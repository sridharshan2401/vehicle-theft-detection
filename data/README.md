# data/

Local input data. Nothing in here is committed to git (footage and registry
embeddings can identify people).

* `samples/` – put test videos here. The smoke tests and benchmarks in
  docs/TESTING.md used `person-bicycle-car-detection.mp4` from Intel's
  [sample-videos](https://github.com/intel-iot-devkit/sample-videos)
  repository (CC BY 4.0). Download it with:

  ```powershell
  Invoke-WebRequest https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4 -OutFile data\samples\person-bicycle-car-detection.mp4
  ```

* `registry/` – appearance embeddings of authorised people, written by
  `python -m app.manage add-person ...`. Delete a file here (and the matching
  registry row) to remove a person.

For training data see [docs/DATASET_GUIDE.md](../docs/DATASET_GUIDE.md).
