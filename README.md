# nezumi-inpaint

ブラウザだけで動く LaMa インペインティングのシングルファイル版デモとライブラリ。
`index.html` はデモページ、`nezumi-inpaint.js` は UMD ライブラリです。

## できること

- 画像のインペイント（LaMa）をブラウザで実行
- Web Worker で推論、IndexedDB キャッシュ対応
- dtype を `float32` / `float16` で切替
- 依存ゼロ（CDN から `onnxruntime-web` を読み込み）

## ファイル構成

- `index.html` デモページ（ワーカーコードを inline で保持）
- `nezumi-inpaint.js` ライブラリ本体（UMD）
- `script.js` 旧デモ用の単体スクリプト（参考）

## 使い方（デモ）

1. ローカルサーバーで `index.html` を開く
2. 画像を読み込み、マスクを塗って「実行」

例（VSCode の Live Server など）:

```bash
# 例: python の簡易サーバー
python3 -m http.server 5500
```

ブラウザで `http://127.0.0.1:5500/index.html` を開きます。

## 使い方（ライブラリ）

```html
<script src="nezumi-inpaint.js"></script>
<script>
  const inpainter = new NezumiInpaint({
    container: '#wrap',
    workerSrc: document.getElementById('workerSrc').textContent,
    dtype: 'float32',
    onStatus: ({ state, text }) => console.log(state, text),
  });

  // File / Blob / HTMLImageElement / URL / dataURL が渡せます
  inpainter.loadImage(file).then(() => inpainter.run());
</script>
```

## モデル URL

現在のデフォルトは Hugging Face です。

- `https://huggingface.co/datasets/Mouserat/nezumi-models/resolve/main/lama_fp32.onnx`

GitHub の旧 URL は廃止されています。

## COOP / COEP 必須

`SharedArrayBuffer` を使うため、以下のヘッダが必要です。

```html
<meta http-equiv="Cross-Origin-Opener-Policy" content="same-origin">
<meta http-equiv="Cross-Origin-Embedder-Policy" content="require-corp">
```

CDN を使う場合は `crossorigin` が必要になることがあります。

## デバッグ

`index.html` のデモはデバッグログを出せます。

```js
window.NEZUMI_DEBUG = true;
```

## クレジット / ライセンス

- LaMa — Apache-2.0 — Resolution-robust Large Mask Inpainting — Suvorov et al., Samsung Research
- onnxruntime-web — MIT — Microsoft
- Space Mono — OFL — Google Fonts
- Noto Sans JP — OFL — Google Fonts
