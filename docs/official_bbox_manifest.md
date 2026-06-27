# Official CAER-S Face BBox Manifest

Patch ini menambahkan dukungan agar manifest CAER-S membaca file deteksi wajah official dari repo reproduksi CAER PyTorch.

Format file deteksi yang didukung:

```text
relative_image_path,label,x1,y1,x2,y2
```

Kolom `label` diabaikan karena label emosi diambil dari folder pada `relative_image_path`, misalnya `Anger/123.png` atau `Happy/456.png`.

## Opsi 1: via CLI

```bash
./bin/build_manifest.sh \
  --config configs/caernet.yaml \
  --train-detections data/caer_official/train.txt \
  --val-detections data/caer_official/val.txt \
  --test-detections data/caer_official/test.txt
```

Jika `--val-detections` diberikan, split validation akan mengikuti file official tersebut. Jika tidak diberikan dan `dataset.create_val_split: true`, validation tetap dibuat dari train set secara stratified holdout.

## Opsi 2: via YAML config

```yaml
dataset:
  dataset_root: /path/to/CAER-S
  image_extensions: [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
  create_val_split: true
  val_ratio: 0.1
  image_size: 224
  detection_files:
    train: data/caer_official/train.txt
    val: data/caer_official/val.txt
    test: data/caer_official/test.txt
```

## Output manifest

Setiap row manifest akan memiliki `face_bbox` terisi:

```json
{
  "sample_id": "test__Happy__123_png",
  "image_path": "test/Happy/123.png",
  "label": "Happy",
  "split": "test",
  "face_bbox": [34, 22, 89, 80]
}
```

Untuk `val`, jika CAER-S tidak memiliki folder `val/`, image path otomatis diarahkan ke `train/<class>/<image>` karena CAER-S official hanya menyediakan folder `train/` dan `test/`.

## Diagnostics

File diagnostics sekarang menambahkan field berikut:

```json
{
  "face_bbox_source": "official_detection_files",
  "face_bbox_counts": {
    "train": 50000,
    "val": 5000,
    "test": 13942
  },
  "val_split_source": "official_detection_file",
  "detection_files": {
    "train": "data/caer_official/train.txt",
    "val": "data/caer_official/val.txt",
    "test": "data/caer_official/test.txt"
  }
}
```

Dengan patch ini, `CAERSTwoStreamDataset` akan memakai bbox official untuk face crop dan face masking pada context branch, sehingga pipeline lebih dekat dengan protokol CAER-S official.
