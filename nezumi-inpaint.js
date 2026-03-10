/**
 * nezumi-inpaint.js
 * LaMa inpainting library — browser, zero-dependency
 *
 * Usage:
 *   const inpainter = new NezumiInpaint({
 *     container: '#canvasWrap',   // CSS selector or HTMLElement
 *     workerSrc: '...',           // worker script source string (or omit to use bundled)
 *     modelUrl:  '...',           // ONNX model URL (optional, defaults to GitHub release)
 *     imgSize:   512,             // inference resolution: 256 / 512 / 768 / 1024
 *     dtype:     'float32',       // 'float32' (default) | 'float16' (half transfer size)
 *     onStatus:  ({ state, text }) => {},
 *     onProgress: ({ pct, label }) => {},
 *     onResult:  ({ elapsedMs, ep }) => {},
 *     onError:   (message) => {},
 *   });
 *
 *   await inpainter.loadImage(file);   // File | Blob | HTMLImageElement | string (URL/dataURL)
 *   inpainter.setMode('mask');         // 'mask' | 'erase'
 *   inpainter.setBrushSize(30);
 *   inpainter.clearMask();
 *   inpainter.undo();
 *   await inpainter.run();             // resolves when inpaint is complete
 *   inpainter.getDtype();              // 'float32' | 'float16'
 *   inpainter.download('result.png');
 *   inpainter.destroy();
 */

(function (root, factory) {
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = factory();
  } else if (typeof define === 'function' && define.amd) {
    define(factory);
  } else {
    root.NezumiInpaint = factory();
  }
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ─── Constants ────────────────────────────────────────────────────────────

  var DEFAULT_MODEL_URL =
    'https://huggingface.co/datasets/Mouserat/nezumi-models/resolve/main/lama_fp32.onnx';
  var DEFAULT_MODEL_URL_FP16 =
    'https://huggingface.co/datasets/Mouserat/nezumi-models/resolve/main/lama_fp16.onnx';

  var DEFAULT_IMG_SIZE = 512;
  var MAX_WIDTH        = 1280;
  var MAX_HEIGHT       = 720;

  /** Valid inference sizes. LaMa was trained on 512; other powers-of-two may work. */
  var VALID_IMG_SIZES = [256, 512, 768, 1024];
  var UNDO_LIMIT  = 20;

  // ─── Helpers ──────────────────────────────────────────────────────────────

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function resolveElement(el) {
    if (typeof el === 'string') return document.querySelector(el);
    return el instanceof HTMLElement ? el : null;
  }

  function noop() {}

  // ─── Float16 utilities ────────────────────────────────────────────────────
  // Pure-JS IEEE 754 fp32 → fp16 packing (no native Float16Array required).
  // Each fp16 value is stored as a uint16 in a Uint16Array.

  /**
   * Encode a Float32Array into a Uint16Array of fp16 bit patterns.
   * @param {Float32Array} f32
   * @returns {Uint16Array}
   */
  function encodeFloat16(f32) {
    var n   = f32.length;
    var u16 = new Uint16Array(n);
    var buf = new ArrayBuffer(4);
    var dv  = new DataView(buf);
    for (var i = 0; i < n; i++) {
      dv.setFloat32(0, f32[i], false);          // big-endian to match bit ops below
      var b = dv.getUint32(0, false);
      var sign  = (b >>> 31) & 0x1;
      var exp32 = (b >>> 23) & 0xff;
      var mant  = b & 0x7fffff;
      var exp16, mant16;

      if (exp32 === 0xff) {
        // NaN or Inf
        exp16  = 0x1f;
        mant16 = mant ? 0x200 : 0;             // preserve NaN-ness
      } else if (exp32 === 0) {
        // Zero / subnormal → zero
        exp16  = 0;
        mant16 = 0;
      } else {
        var e = exp32 - 127 + 15;              // re-bias
        if (e >= 31) {
          // Overflow → Inf
          exp16  = 0x1f;
          mant16 = 0;
        } else if (e <= 0) {
          // Underflow → subnormal fp16 or zero
          if (e < -10) {
            exp16  = 0;
            mant16 = 0;
          } else {
            // Subnormal: shift mantissa
            exp16  = 0;
            mant16 = (mant | 0x800000) >>> (14 - exp32 + 127);
          }
        } else {
          exp16  = e;
          mant16 = mant >>> 13;
        }
      }
      u16[i] = (sign << 15) | (exp16 << 10) | mant16;
    }
    return u16;
  }

  /**
   * Decode a Uint16Array of fp16 bit patterns back to Float32Array.
   * Used when the worker returns fp16 results.
   * @param {Uint16Array} u16
   * @returns {Float32Array}
   */
  function decodeFloat16(u16) {
    var n   = u16.length;
    var f32 = new Float32Array(n);
    for (var i = 0; i < n; i++) {
      var h     = u16[i];
      var sign  = (h >>> 15) ? -1 : 1;
      var exp16 = (h >>> 10) & 0x1f;
      var mant  =  h         & 0x3ff;
      if (exp16 === 0) {
        f32[i] = sign * Math.pow(2, -14) * (mant / 1024);   // subnormal
      } else if (exp16 === 0x1f) {
        f32[i] = mant ? NaN : sign * Infinity;
      } else {
        f32[i] = sign * Math.pow(2, exp16 - 15) * (1 + mant / 1024);
      }
    }
    return f32;
  }

  // ─── NezumiInpaint ────────────────────────────────────────────────────────

  /**
   * @param {object} opts
   * @param {string|HTMLElement} opts.container
   * @param {string}  [opts.workerSrc]
   * @param {string}  [opts.modelUrl]
   * @param {number}  [opts.brushSize=24]
   * @param {number}  [opts.undoLimit=20]
   * @param {number}  [opts.imgSize=512]  Inference resolution. Snapped to nearest of 256/512/768/1024.
   * @param {'float32'|'float16'} [opts.dtype='float32']  Tensor dtype sent to the worker.
   *   'float16' halves transfer size but requires the worker/model to accept fp16 input.
   *   Falls back to float32 automatically if Float16Array / encodeFloat16 is unavailable.
   * @param {boolean} [opts.preferWebGPU=true]
   * @param {function} [opts.onStatus]
   * @param {function} [opts.onProgress]
   * @param {function} [opts.onResult]
   * @param {function} [opts.onError]
   */
  function NezumiInpaint(opts) {
    opts = opts || {};

    // ── Config ──────────────────────────────────────────────────────────────
    this._workerSrc     = opts.workerSrc  || null;
    this._preferWebGPU  = opts.preferWebGPU !== false;
    this._undoLimit     = opts.undoLimit  || UNDO_LIMIT;
    this._brushSize     = clamp(opts.brushSize || 24, 2, 200);
    this._mode          = 'mask';        // 'mask' | 'erase'

    // imgSize: snapped to nearest valid value (default 512)
    var reqSize = opts.imgSize || DEFAULT_IMG_SIZE;
    this._imgSize = VALID_IMG_SIZES.reduce(function (best, s) {
      return Math.abs(s - reqSize) < Math.abs(best - reqSize) ? s : best;
    }, DEFAULT_IMG_SIZE);

    // dtype: 'float32' (default) or 'float16'
    var wantF16 = (opts.dtype === 'float16');
    this._useFloat16    = wantF16;
    this._nativeFloat16 = wantF16 && (typeof Float16Array !== 'undefined');

    // modelUrl: always fp32 model — dtype:'float16' affects transfer only, not the model
    this._modelUrl = opts.modelUrl || DEFAULT_MODEL_URL;

    // ── Callbacks ────────────────────────────────────────────────────────────
    this._onStatus   = opts.onStatus   || noop;
    this._onProgress = opts.onProgress || noop;
    this._onResult   = opts.onResult   || noop;
    this._onError    = opts.onError    || noop;

    // ── State ────────────────────────────────────────────────────────────────
    this._workerReady = false;
    this._imageLoaded = false;
    this._painting    = false;
    this._undoStack   = [];
    this._worker      = null;
    this._lastEP      = 'unknown';
    this._cachedImgArr = null;
    this._pendingResolve = null;
    this._pendingReject  = null;

    // ── DOM / Canvases ───────────────────────────────────────────────────────
    this._container  = resolveElement(opts.container);
    this._imgCanvas  = null;
    this._maskCanvas = null;
    this._curCanvas  = null;
    this._ptrLayer   = null;
    this._imgCtx     = null;
    this._maskCtx    = null;
    this._curCtx     = null;

    // Off-screen canvases sized to inference resolution
    this._imgSmall  = document.createElement('canvas');
    this._maskSmall = document.createElement('canvas');
    this._imgSmall.width  = this._maskSmall.width  = this._imgSize;
    this._imgSmall.height = this._maskSmall.height = this._imgSize;
    this._imgSmallCtx  = this._imgSmall.getContext('2d', { willReadFrequently: true });
    this._maskSmallCtx = this._maskSmall.getContext('2d', { willReadFrequently: true });

    if (this._container) this._buildDOM();
    this._startWorker(this._preferWebGPU);
  }

  // ── DOM helpers ─────────────────────────────────────────────────────────

  NezumiInpaint.prototype._buildDOM = function () {
    var self = this;
    var root = this._container;
    root.style.position = 'relative';

    function makeCanvas(id) {
      var c = document.createElement('canvas');
      c.id = id;
      c.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;';
      root.appendChild(c);
      return c;
    }

    this._imgCanvas  = makeCanvas('ni-imageCanvas');
    this._maskCanvas = makeCanvas('ni-maskCanvas');
    this._curCanvas  = makeCanvas('ni-cursorCanvas');
    this._ptrLayer   = makeCanvas('ni-pointerLayer');
    this._ptrLayer.style.cursor = 'none';
    this._ptrLayer.style.zIndex = '10';

    this._imgCtx  = this._imgCanvas.getContext('2d',  { willReadFrequently: true });
    this._maskCtx = this._maskCanvas.getContext('2d', { willReadFrequently: true });
    this._curCtx  = this._curCanvas.getContext('2d');

    this._attachPointerEvents();
  };

  NezumiInpaint.prototype._resizeCanvases = function (w, h) {
    var canvases = [this._imgCanvas, this._maskCanvas, this._curCanvas, this._ptrLayer];
    canvases.forEach(function (c) { if (c) { c.width = w; c.height = h; } });
  };

  // ── Pointer / brush events ───────────────────────────────────────────────

  NezumiInpaint.prototype._attachPointerEvents = function () {
    var self = this;
    var ptr  = this._ptrLayer;

    ptr.addEventListener('pointermove', function (e) {
      var pos = self._getPos(e);
      self._paintCursor(pos);
      if (self._painting) self._paintBrush(pos);
    });

    ptr.addEventListener('pointerdown', function (e) {
      if (!self._imageLoaded) return;
      self._painting = true;
      ptr.setPointerCapture(e.pointerId);
      self._paintBrush(self._getPos(e));
    });

    ptr.addEventListener('pointerup',    function ()  { self._painting = false; });
    ptr.addEventListener('pointerleave', function ()  {
      self._painting = false;
      self._curCtx.clearRect(0, 0, self._curCanvas.width, self._curCanvas.height);
    });
  };

  NezumiInpaint.prototype._getPos = function (e) {
    var rect = this._ptrLayer.getBoundingClientRect();
    var sx   = this._imgCanvas.width  / rect.width;
    var sy   = this._imgCanvas.height / rect.height;
    var src  = e.touches ? e.touches[0] : e;
    return { x: (src.clientX - rect.left) * sx, y: (src.clientY - rect.top) * sy };
  };

  NezumiInpaint.prototype._paintBrush = function (pos) {
    var r   = this._brushSize / 2;
    var ctx = this._maskCtx;
    ctx.globalCompositeOperation = this._mode === 'mask' ? 'source-over' : 'destination-out';
    ctx.fillStyle = this._mode === 'mask' ? 'rgba(255,60,100,0.9)' : 'rgba(0,0,0,1)';
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalCompositeOperation = 'source-over';
  };

  NezumiInpaint.prototype._paintCursor = function (pos) {
    if (!this._imageLoaded) return;
    var r   = this._brushSize / 2;
    var col = this._mode === 'mask' ? '#ff4f7b' : '#00e5a0';
    var ctx = this._curCtx;
    ctx.clearRect(0, 0, this._curCanvas.width, this._curCanvas.height);
    ctx.strokeStyle = col;
    ctx.lineWidth   = 1.5;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pos.x - 5, pos.y); ctx.lineTo(pos.x + 5, pos.y);
    ctx.moveTo(pos.x, pos.y - 5); ctx.lineTo(pos.x, pos.y + 5);
    ctx.stroke();
  };

  // ── Image loading ────────────────────────────────────────────────────────

  /**
   * Load an image into the inpainter.
   * @param {File|Blob|HTMLImageElement|string} source  File/Blob, img element, URL, or dataURL
   * @returns {Promise<{width:number, height:number}>}
   */
  NezumiInpaint.prototype.loadImage = function (source) {
    var self = this;
    return new Promise(function (resolve, reject) {
      function onImage(img) {
        var w = img.naturalWidth  || img.width;
        var h = img.naturalHeight || img.height;
        if (w > MAX_WIDTH)  { h = Math.round(h * MAX_WIDTH  / w); w = MAX_WIDTH;  }
        if (h > MAX_HEIGHT) { w = Math.round(w * MAX_HEIGHT / h); h = MAX_HEIGHT; }

        self._resizeCanvases(w, h);
        if (self._container) self._container.style.aspectRatio = w + '/' + h;
        self._imgCtx.drawImage(img, 0, 0, w, h);
        self._maskCtx.clearRect(0, 0, w, h);
        self._imageLoaded = true;
        self._undoStack.length = 0;
        self._rebuildImageCache();
        self._onStatus({ state: 'ok', text: w + ' × ' + h + ' px loaded' });
        resolve({ width: w, height: h });
      }

      function fromURL(url) {
        var img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload  = function () { onImage(img); };
        img.onerror = function () { reject(new Error('Image decode failed')); };
        img.src = url;
      }

      if (source instanceof HTMLImageElement) {
        if (source.complete) onImage(source);
        else { source.onload = function () { onImage(source); }; }
      } else if (source instanceof File || source instanceof Blob) {
        var reader = new FileReader();
        reader.onload  = function (ev) { fromURL(ev.target.result); };
        reader.onerror = function ()   { reject(new Error('File read failed')); };
        reader.readAsDataURL(source);
      } else if (typeof source === 'string') {
        fromURL(source);
      } else {
        reject(new Error('Unsupported source type'));
      }
    });
  };

  // ── Tensor cache ─────────────────────────────────────────────────────────

  NezumiInpaint.prototype._rebuildImageCache = function () {
    if (!this._imageLoaded) return;
    var S   = this._imgSize;
    var ctx = this._imgSmallCtx;
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, S, S);
    ctx.drawImage(this._imgCanvas, 0, 0, S, S);
    var data = ctx.getImageData(0, 0, S, S).data;
    var n = S * S;

    // Always build an intermediate Float32Array for accuracy, then optionally
    // convert to fp16 (Uint16Array) for the cached tensor.
    var f32 = new Float32Array(3 * n);
    for (var i = 0; i < n; i++) {
      f32[0 * n + i] = data[i * 4 + 0] / 255;
      f32[1 * n + i] = data[i * 4 + 1] / 255;
      f32[2 * n + i] = data[i * 4 + 2] / 255;
    }

    // Always cache as Float32Array.
    // dtype:'float16' affects transfer encoding in run(), not the cache itself.
    this._cachedImgArr = f32;
  };

  // ── Worker ───────────────────────────────────────────────────────────────

  NezumiInpaint.prototype._startWorker = function (preferGPU) {
    var self = this;
    if (this._worker) {
      try { this._worker.terminate(); } catch (e) { /* ignore */ }
    }

    var src, url;
    if (this._workerSrc) {
      var blob = new Blob([this._workerSrc], { type: 'application/javascript' });
      url = URL.createObjectURL(blob);
    } else {
      // Fallback: look for an inline <script id="workerSrc"> element (legacy support)
      var el = document.getElementById('workerSrc');
      if (el) {
        var blob2 = new Blob([el.textContent], { type: 'application/javascript' });
        url = URL.createObjectURL(blob2);
      } else {
        this._onError('No worker source provided. Pass workerSrc option.');
        return;
      }
    }

    this._worker = new Worker(url);

    this._worker.onmessage = function (e) {
      var msg = e.data;
      switch (msg.type) {
        case 'status':
          self._onStatus({ state: msg.state, text: msg.text });
          break;

        case 'progress':
          self._onProgress({ pct: msg.pct, label: msg.label });
          break;

        case 'ready':
          self._workerReady = true;
          self._lastEP = msg.ep || 'wasm';
          self._onStatus({ state: 'ok', text: 'LaMa ready (' + self._lastEP + ')' });
          self._onProgress({ pct: 100, label: '' });
          break;

        case 'result':
          self._handleResult(msg);
          break;

        case 'error':
          self._onStatus({ state: 'err', text: 'Error: ' + msg.text });
          self._onError(msg.text);
          if (self._undoStack.length) {
            self._imgCtx.putImageData(self._undoStack.pop(), 0, 0);
            self._rebuildImageCache();
          }
          if (self._pendingReject) {
            self._pendingReject(new Error(msg.text));
            self._pendingResolve = self._pendingReject = null;
          }
          break;
      }
    };

    this._worker.onerror = function (e) {
      self._onError('Worker error: ' + e.message);
    };

    this._worker.postMessage({
      type:         'init',
      modelUrl:     this._modelUrl,
      threads:      navigator.hardwareConcurrency || 2,
      preferWebGPU: !!preferGPU
    });
  };

  NezumiInpaint.prototype._handleResult = function (msg) {
    var S   = this._imgSize;
    var W   = this._imgCanvas.width;
    var H   = this._imgCanvas.height;

    function debugRgba(label, src, stats) {
      if (typeof window === 'undefined' || !window.NEZUMI_DEBUG) return;
      var len = src && src.length !== undefined ? src.length : (src && src.byteLength) || 0;
      var ctor = src && src.constructor ? src.constructor.name : typeof src;
      var sample = [];
      try {
        var arr = (src instanceof ArrayBuffer) ? new Uint8Array(src, 0, Math.min(16, src.byteLength)) :
                  (src instanceof Uint8Array || src instanceof Uint8ClampedArray) ? src :
                  (src instanceof Float32Array || src instanceof Float64Array) ? src :
                  Array.isArray(src) ? src : null;
        if (arr) {
          var n = Math.min(16, arr.length);
          for (var i = 0; i < n; i++) sample.push(arr[i]);
        }
      } catch (e) {}
      console.log('[NezumiInpaint]', label, { ctor: ctor, len: len, sample: sample, ep: msg.ep, hasF16: !!msg.rgbaF16, stats: stats || null });
    }

    function floatToU8(arr, n) {
      var out = new Uint8ClampedArray(n);
      var max = 0;
      for (var i = 0; i < n; i++) if (arr[i] > max) max = arr[i];
      var scale = (max <= 1.01) ? 255 : 1;
      for (var j = 0; j < n; j++) out[j] = arr[j] * scale;
      return out;
    }

    function chwToRgba(chw, nPix) {
      var out = new Uint8ClampedArray(nPix * 4);
      var max = 0;
      for (var i = 0; i < chw.length; i++) if (chw[i] > max) max = chw[i];
      var scale = (max <= 1.01) ? 255 : 1;
      for (var p = 0; p < nPix; p++) {
        out[p * 4 + 0] = chw[p] * scale;
        out[p * 4 + 1] = chw[nPix + p] * scale;
        out[p * 4 + 2] = chw[2 * nPix + p] * scale;
        out[p * 4 + 3] = 255;
      }
      return out;
    }

    function normalizeRgba(src) {
      var nPix = S * S;
      var n = nPix * 4;
      if (!src) return new Uint8ClampedArray(n);
      if (src instanceof Uint8ClampedArray) {
        if (src.length === n) return src;
        if (src.length === nPix * 3) return chwToRgba(src, nPix);
      }
      if (src instanceof Uint8Array) {
        if (src.length === n) return new Uint8ClampedArray(src.buffer, src.byteOffset, n);
        if (src.length === nPix * 3) return chwToRgba(src, nPix);
      }
      if (src instanceof Float32Array || src instanceof Float64Array) {
        if (src.length === nPix * 3) return chwToRgba(src, nPix);
        return floatToU8(src, n);
      }
      if (src instanceof ArrayBuffer) {
        if (src.byteLength === n * 4) return floatToU8(new Float32Array(src), n);
        if (src.byteLength === nPix * 3 * 4) return chwToRgba(new Float32Array(src), nPix);
        return new Uint8ClampedArray(src);
      }
      if (Array.isArray(src)) {
        if (src.length === nPix * 3) return chwToRgba(src, nPix);
        return floatToU8(src, n);
      }
      return new Uint8ClampedArray(n);
    }

    // Worker may return rgba as Uint8ClampedArray (fp32 path) or
    // Uint16Array (fp16 path — values are fp16 bit patterns, 0..1 range).
    var rgba;
    if (msg.rgbaF16) {
      // fp16 result: decode each channel back to [0,255] uint8
      var f32ch = decodeFloat16(
        msg.rgbaF16 instanceof Uint16Array ? msg.rgbaF16 : new Uint16Array(msg.rgbaF16)
      );
      rgba = new Uint8ClampedArray(f32ch.length);
      for (var k = 0; k < f32ch.length; k++) rgba[k] = f32ch[k] * 255;
    } else {
      debugRgba('raw rgba', msg.rgba, msg.stats);
      rgba = normalizeRgba(msg.rgba);
      debugRgba('normalized rgba', rgba, msg.stats);
    }
    var tmp = document.createElement('canvas');
    tmp.width = S; tmp.height = S;
    tmp.getContext('2d').putImageData(new ImageData(rgba, S, S), 0, 0);
    this._imgCtx.drawImage(tmp, 0, 0, W, H);
    this._rebuildImageCache();
    this._maskCtx.clearRect(0, 0, W, H);

    var ep      = msg.ep || this._lastEP || 'unknown';
    var elapsed = typeof msg.elapsedMs === 'number' ? msg.elapsedMs.toFixed(1) : '?';
    this._lastEP = ep;
    this._onStatus({ state: 'ok', text: 'Done (' + ep + ', ' + elapsed + ' ms)' });
    this._onResult({ elapsedMs: msg.elapsedMs, ep: ep });

    // Auto-fallback: WebGPU too slow → switch to WASM
    if (ep === 'webgpu' && typeof msg.elapsedMs === 'number' && msg.elapsedMs > 15000 && this._preferWebGPU) {
      this._preferWebGPU = false;
      this._workerReady  = false;
      this._onStatus({ state: 'busy', text: 'WebGPU slow — switching to WASM…' });
      this._startWorker(false);
    }

    if (this._pendingResolve) {
      this._pendingResolve({ elapsedMs: msg.elapsedMs, ep: ep });
      this._pendingResolve = this._pendingReject = null;
    }
  };

  // ── Public API ───────────────────────────────────────────────────────────

  /**
   * Set drawing mode.
   * @param {'mask'|'erase'} m
   */
  NezumiInpaint.prototype.setMode = function (m) {
    if (m === 'mask' || m === 'erase') this._mode = m;
  };

  /**
   * Get current drawing mode.
   * @returns {'mask'|'erase'}
   */
  NezumiInpaint.prototype.getMode = function () { return this._mode; };

  /**
   * Set brush size in pixels.
   * @param {number} size  2–200
   */
  NezumiInpaint.prototype.setBrushSize = function (size) {
    this._brushSize = clamp(size, 2, 200);
  };

  /** @returns {number} */
  NezumiInpaint.prototype.getBrushSize = function () { return this._brushSize; };

  /** Clear the mask canvas. */
  NezumiInpaint.prototype.clearMask = function () {
    if (this._maskCtx) {
      this._maskCtx.clearRect(0, 0, this._maskCanvas.width, this._maskCanvas.height);
    }
  };

  /**
   * Undo the last inpaint result.
   * @returns {boolean}  true if undo was available
   */
  NezumiInpaint.prototype.undo = function () {
    if (!this._undoStack.length) return false;
    this._imgCtx.putImageData(this._undoStack.pop(), 0, 0);
    this._rebuildImageCache();
    this.clearMask();
    this._onStatus({ state: 'ok', text: 'Undo (' + this._undoStack.length + ' left)' });
    return true;
  };

  /** @returns {number} number of undo steps available */
  NezumiInpaint.prototype.undoCount = function () { return this._undoStack.length; };

  /**
   * Run LaMa inpainting.  Resolves when the result is applied to the canvas.
   * @returns {Promise<{elapsedMs:number, ep:string}>}
   */
  NezumiInpaint.prototype.run = function () {
    var self = this;

    if (!this._workerReady) return Promise.reject(new Error('Model not ready'));
    if (!this._imageLoaded) return Promise.reject(new Error('No image loaded'));

    // Build mask tensor (always fp32 first; convert if needed)
    var S = this._imgSize;
    var n = S * S;
    if (!this._cachedImgArr) this._rebuildImageCache();
    var imgArr = this._cachedImgArr.slice();  // detach for zero-copy transfer

    var mCtx = this._maskSmallCtx;
    mCtx.imageSmoothingEnabled = false;
    mCtx.clearRect(0, 0, S, S);
    mCtx.drawImage(this._maskCanvas, 0, 0, S, S);
    var mData = mCtx.getImageData(0, 0, S, S).data;

    var maskF32 = new Float32Array(n);
    var ones = 0;
    for (var j = 0; j < n; j++) {
      var v = mData[j * 4 + 3] > 10 ? 1.0 : 0.0;
      maskF32[j] = v;
      if (v) ones++;
    }

    if (!ones) return Promise.reject(new Error('Mask is empty — draw on the image first'));

    // Encode to fp16 for transfer if requested
    var sendImg, sendMask, dtype;
    if (this._useFloat16) {
      sendImg  = encodeFloat16(imgArr);   // Uint16Array
      sendMask = encodeFloat16(maskF32);  // Uint16Array
      dtype    = 'float16';
    } else {
      sendImg  = imgArr;    // Float32Array
      sendMask = maskF32;   // Float32Array
      dtype    = 'float32';
    }

    // Push undo snapshot (capped)
    var snap = this._imgCtx.getImageData(0, 0, this._imgCanvas.width, this._imgCanvas.height);
    if (this._undoStack.length >= this._undoLimit) this._undoStack.shift();
    this._undoStack.push(snap);

    this._onStatus({ state: 'busy', text: 'Running LaMa…' });

    return new Promise(function (resolve, reject) {
      self._pendingResolve = resolve;
      self._pendingReject  = reject;
      // imgArr は slice() 済みの独立バッファ、maskArr も新規作成 → 両方 transfer 可
      var imgBuf  = sendImg.buffer;
      var maskBuf = sendMask.buffer;
      self._worker.postMessage(
        { type: 'run', imgArr: imgBuf, maskArr: maskBuf, dtype: dtype },
        [imgBuf, maskBuf]
      );
      // transfer 後に _cachedImgArr が detach されないよう null にして次回再構築を強制
      self._cachedImgArr = null;
    });
  };

  /**
   * Export the current (inpainted) image as a PNG data-URL.
   * @returns {string}
   */
  NezumiInpaint.prototype.toDataURL = function () {
    return this._imgCanvas ? this._imgCanvas.toDataURL('image/png') : '';
  };

  /**
   * Trigger a PNG download of the current image.
   * @param {string} [filename='inpaint-result.png']
   */
  NezumiInpaint.prototype.download = function (filename) {
    var url = this.toDataURL();
    if (!url) return;
    var a   = document.createElement('a');
    a.href  = url;
    a.download = filename || 'inpaint-result.png';
    a.click();
  };

  /** @returns {number} Current inference resolution (e.g. 256, 512, 768, 1024). */
  NezumiInpaint.prototype.getImgSize = function () { return this._imgSize; };

  /**
   * Returns the active tensor dtype used for worker communication.
   * @returns {'float32'|'float16'}
   */
  NezumiInpaint.prototype.getDtype = function () {
    return this._useFloat16 ? 'float16' : 'float32';
  };

  /**
   * Returns true when the model worker is ready to accept run() calls.
   * @returns {boolean}
   */
  NezumiInpaint.prototype.isReady = function () { return this._workerReady; };

  /**
   * Returns true when an image has been loaded.
   * @returns {boolean}
   */
  NezumiInpaint.prototype.hasImage = function () { return this._imageLoaded; };

  /**
   * Destroy the instance: terminate the worker and remove canvases.
   */
  NezumiInpaint.prototype.destroy = function () {
    if (this._worker) { try { this._worker.terminate(); } catch (e) { /* ignore */ } }
    if (this._container) {
      var canvases = this._container.querySelectorAll('canvas[id^="ni-"]');
      canvases.forEach(function (c) { c.parentNode.removeChild(c); });
    }
    this._worker = null;
    this._workerReady = false;
    this._imageLoaded = false;
  };

  return NezumiInpaint;
}));
