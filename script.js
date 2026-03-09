var MODEL_URL = './lama_fp32.onnx';
var IMG_SIZE  = 512;
var lastEP    = 'unknown';
var preferWebGPU = true;

// ---- state ----
var workerReady = false;
var imageLoaded = false;
var painting    = false;
var mode        = 'mask';
var undoStack   = [];
var worker      = null;

// ---- elements ----
var canvasWrap  = document.getElementById('canvasWrap');
var imgCanvas   = document.getElementById('imageCanvas');
var maskCanvas  = document.getElementById('maskCanvas');
var curCanvas   = document.getElementById('cursorCanvas');
var ptrLayer    = document.getElementById('pointerLayer');
var dropOverlay = document.getElementById('dropOverlay');
var fileInput   = document.getElementById('fileInput');
var btnRun      = document.getElementById('btnRun');
var btnClear    = document.getElementById('btnClearMask');
var btnUndo     = document.getElementById('btnUndo');
var btnOpen     = document.getElementById('btnOpen');
var btnMaskEl   = document.getElementById('btnMask');
var btnEraseEl  = document.getElementById('btnErase');
var brushSlider = document.getElementById('brushSize');
var sizeVal     = document.getElementById('sizeVal');
var statusDot   = document.getElementById('statusDot');
var statusText  = document.getElementById('statusText');
var progressWrap  = document.getElementById('progressWrap');
var progressFill  = document.getElementById('progressFill');
var progressLabel = document.getElementById('progressLabel');

var imgCtx  = imgCanvas.getContext('2d', { willReadFrequently:true });
var maskCtx = maskCanvas.getContext('2d', { willReadFrequently:true });
var curCtx  = curCanvas.getContext('2d');
var imgSmallCanvas = document.createElement('canvas');
var maskSmallCanvas = document.createElement('canvas');
imgSmallCanvas.width = IMG_SIZE;
imgSmallCanvas.height = IMG_SIZE;
maskSmallCanvas.width = IMG_SIZE;
maskSmallCanvas.height = IMG_SIZE;
var imgSmallCtx = imgSmallCanvas.getContext('2d', { willReadFrequently:true });
var maskSmallCtx = maskSmallCanvas.getContext('2d', { willReadFrequently:true });
var cachedImgArr = null;

function setStatus(state, text) {
  statusDot.className = 'dot ' + state;
  statusText.textContent = text;
}
function setProgress(pct, label) {
  progressWrap.classList.toggle('visible', pct < 100);
  progressFill.style.width = pct + '%';
  progressLabel.textContent = label;
}
function resizeCanvases(w, h) {
  [imgCanvas, maskCanvas, curCanvas].forEach(function(c){ c.width=w; c.height=h; });
}
function rebuildImageTensorCache() {
  if (!imageLoaded) return;
  imgSmallCtx.imageSmoothingEnabled = true;
  imgSmallCtx.clearRect(0, 0, IMG_SIZE, IMG_SIZE);
  imgSmallCtx.drawImage(imgCanvas, 0, 0, IMG_SIZE, IMG_SIZE);
  var imgD = imgSmallCtx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data;
  var n = IMG_SIZE * IMG_SIZE;
  if (!cachedImgArr || cachedImgArr.length !== 3 * n) {
    cachedImgArr = new Float32Array(3 * n);
  }
  for (var i = 0; i < n; i++) {
    cachedImgArr[0*n+i] = imgD[i*4+0] / 255;
    cachedImgArr[1*n+i] = imgD[i*4+1] / 255;
    cachedImgArr[2*n+i] = imgD[i*4+2] / 255;
  }
}

// ---- Worker ----
function startWorker(preferGPU) {
  if (worker) {
    try { worker.terminate(); } catch (e) {}
  }
  var src  = document.getElementById('workerSrc').textContent;
  var blob = new Blob([src], { type: 'application/javascript' });
  var url  = URL.createObjectURL(blob);
  worker   = new Worker(url);

  worker.onmessage = function(e) {
    var msg = e.data;
    if (msg.type === 'status')   { setStatus(msg.state, msg.text); }
    if (msg.type === 'progress') { setProgress(msg.pct, msg.label); }
    if (msg.type === 'ready') {
      workerReady = true;
      var epLabel = msg.ep || 'wasm';
      lastEP = epLabel;
      setStatus('ok', 'LaMa ready (' + epLabel + ') — 画像を読み込んでください');
      setProgress(100, '');
      if (imageLoaded) btnRun.disabled = false;
    }
    if (msg.type === 'error') {
      setStatus('err', 'エラー: ' + msg.text);
      btnRun.disabled = false;
      if (undoStack.length) imgCtx.putImageData(undoStack.pop(), 0, 0);
    }
      if (msg.type === 'result') {
      var W = imgCanvas.width, H = imgCanvas.height;
      var tmp = document.createElement('canvas');
      tmp.width = IMG_SIZE; tmp.height = IMG_SIZE;
      tmp.getContext('2d').putImageData(new ImageData(msg.rgba, IMG_SIZE, IMG_SIZE), 0, 0);
      imgCtx.drawImage(tmp, 0, 0, W, H);
      rebuildImageTensorCache();
      maskCtx.clearRect(0, 0, W, H);
      var epUsed = msg.ep || lastEP || 'unknown';
      var elapsed = (typeof msg.elapsedMs === 'number') ? msg.elapsedMs.toFixed(1) : '?';
      setStatus('ok', 'LaMa inpaint 完了 (' + epUsed + ', ' + elapsed + ' ms)');
      btnRun.disabled = false;

      // Some environments report webgpu but run very slowly.
      // If one run is extremely slow, switch to wasm for following runs.
      if (epUsed === 'webgpu' && typeof msg.elapsedMs === 'number' && msg.elapsedMs > 15000 && preferWebGPU) {
        preferWebGPU = false;
        workerReady = false;
        btnRun.disabled = true;
        setStatus('busy', 'webgpu が遅いため wasm に切替中…');
        startWorker(false);
      }
    }
  };

  worker.onerror = function(e) {
    setStatus('err', 'Worker エラー: ' + e.message);
  };

  worker.postMessage({
    type: 'init',
    modelUrl: MODEL_URL,
    threads: navigator.hardwareConcurrency || 2,
    preferWebGPU: !!preferGPU
  });
}

startWorker(preferWebGPU);

// ---- image loading ----
function loadFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) { setStatus('err', '画像ファイルを選択してください'); return; }
  setStatus('busy', '読み込み中…');
  var reader = new FileReader();
  reader.onload = function(ev) {
    var img = new Image();
    img.onload = function() {
      var w = img.naturalWidth, h = img.naturalHeight;
      if (w > 1280) { h = Math.round(h*1280/w); w = 1280; }
      if (h > 720)  { w = Math.round(w*720/h);  h = 720;  }
      resizeCanvases(w, h);
      canvasWrap.style.aspectRatio = w+'/'+h;
      imgCtx.drawImage(img, 0, 0, w, h);
      maskCtx.clearRect(0, 0, w, h);
      dropOverlay.classList.add('hidden');
      imageLoaded = true;
      undoStack.length = 0;
      rebuildImageTensorCache();
      if (workerReady) btnRun.disabled = false;
      setStatus('ok', w+' x '+h+' px 読み込み完了');
    };
    img.onerror = function(){ setStatus('err', '画像のデコードに失敗しました'); };
    img.src = ev.target.result;
  };
  reader.onerror = function(){ setStatus('err', 'ファイルの読み取りに失敗しました'); };
  reader.readAsDataURL(file);
}
fileInput.addEventListener('change', function(){ loadFile(fileInput.files[0]); fileInput.value=''; });
function openPicker(){ fileInput.click(); }
btnOpen.addEventListener('click', openPicker);
dropOverlay.addEventListener('click', openPicker);
canvasWrap.addEventListener('dragover', function(e){ e.preventDefault(); });
canvasWrap.addEventListener('drop', function(e){ e.preventDefault(); loadFile(e.dataTransfer.files[0]); });

// ---- sliders / mode ----
brushSlider.addEventListener('input', function(){ sizeVal.textContent = brushSlider.value; });
btnMaskEl.addEventListener('click',  function(){ setMode('mask'); });
btnEraseEl.addEventListener('click', function(){ setMode('erase'); });
function setMode(m) {
  mode = m;
  btnMaskEl.classList.toggle('active',  m==='mask');
  btnEraseEl.classList.toggle('active', m==='erase');
}

// ---- brush / cursor ----
function getPos(e) {
  var rect = ptrLayer.getBoundingClientRect();
  var sx = imgCanvas.width/rect.width, sy = imgCanvas.height/rect.height;
  var src = e.touches ? e.touches[0] : e;
  return { x:(src.clientX-rect.left)*sx, y:(src.clientY-rect.top)*sy };
}
function paintBrush(pos) {
  var r = parseInt(brushSlider.value)/2;
  maskCtx.globalCompositeOperation = mode==='mask' ? 'source-over' : 'destination-out';
  maskCtx.fillStyle = mode==='mask' ? 'rgba(255,60,100,0.9)' : 'rgba(0,0,0,1)';
  maskCtx.beginPath(); maskCtx.arc(pos.x,pos.y,r,0,Math.PI*2); maskCtx.fill();
  maskCtx.globalCompositeOperation = 'source-over';
}
function paintCursor(pos) {
  if (!imageLoaded) return;
  var r = parseInt(brushSlider.value)/2;
  var col = mode==='mask' ? '#ff4f7b' : '#00e5a0';
  curCtx.clearRect(0,0,curCanvas.width,curCanvas.height);
  curCtx.strokeStyle=col; curCtx.lineWidth=1.5;
  curCtx.beginPath(); curCtx.arc(pos.x,pos.y,r,0,Math.PI*2); curCtx.stroke();
  curCtx.lineWidth=1;
  curCtx.beginPath();
  curCtx.moveTo(pos.x-5,pos.y); curCtx.lineTo(pos.x+5,pos.y);
  curCtx.moveTo(pos.x,pos.y-5); curCtx.lineTo(pos.x,pos.y+5);
  curCtx.stroke();
}
ptrLayer.addEventListener('pointermove',  function(e){ paintCursor(getPos(e)); if(painting) paintBrush(getPos(e)); });
ptrLayer.addEventListener('pointerdown',  function(e){ if(!imageLoaded)return; painting=true; ptrLayer.setPointerCapture(e.pointerId); paintBrush(getPos(e)); });
ptrLayer.addEventListener('pointerup',    function(){ painting=false; });
ptrLayer.addEventListener('pointerleave', function(){ painting=false; curCtx.clearRect(0,0,curCanvas.width,curCanvas.height); });

// ---- clear / undo ----
btnClear.addEventListener('click', function(){ maskCtx.clearRect(0,0,maskCanvas.width,maskCanvas.height); });
btnUndo.addEventListener('click', function(){
  if (!undoStack.length) return;
  imgCtx.putImageData(undoStack.pop(),0,0);
  rebuildImageTensorCache();
  maskCtx.clearRect(0,0,maskCanvas.width,maskCanvas.height);
  setStatus('ok','undo (残り '+undoStack.length+')');
});

// ---- run ----
btnRun.addEventListener('click', function() {
  if (!workerReady)  { setStatus('err','モデルがまだ準備できていません'); return; }
  if (!imageLoaded)  { setStatus('err','画像を読み込んでください'); return; }

  undoStack.push(imgCtx.getImageData(0,0,imgCanvas.width,imgCanvas.height));
  btnRun.disabled = true;
  setStatus('busy','前処理中…');
  var n = IMG_SIZE * IMG_SIZE;
  if (!cachedImgArr) rebuildImageTensorCache();
  var imgArr = cachedImgArr.slice();

  // mask -> float32 (alpha-based binary mask)
  maskSmallCtx.imageSmoothingEnabled = false;
  maskSmallCtx.clearRect(0, 0, IMG_SIZE, IMG_SIZE);
  maskSmallCtx.drawImage(maskCanvas, 0, 0, IMG_SIZE, IMG_SIZE);
  var mskD = maskSmallCtx.getImageData(0,0,IMG_SIZE,IMG_SIZE).data;
  var maskArr = new Float32Array(n);
  var ones = 0;
  for (var j=0; j<n; j++) {
    var alpha = mskD[j*4 + 3];
    var v = alpha > 10 ? 1.0 : 0.0;
    maskArr[j] = v;
    if (v === 1.0) ones++;
  }
  if (!ones) { setStatus('err','マスクを描いてから実行してください'); btnRun.disabled = false; undoStack.pop(); return; }

  // transfer to worker (zero-copy)
  worker.postMessage({ type:'run', imgArr:imgArr, maskArr:maskArr }, [imgArr.buffer, maskArr.buffer]);
});
