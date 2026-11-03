# Licensing and provenance

Copyright 2026 VidUEU contributors. Original VidUEU source code and documentation
are licensed under Apache-2.0; see LICENSE. This grant does not relicense
third-party code, model weights, datasets or generated media.

`vendor/bagel` is a customized BAGEL training snapshot. Its Apache-2.0 license
is preserved in `vendor/bagel/LICENSE`; `SOURCE_MANIFEST.json` records the
original file fingerprints and source commit. Release-specific changes to
that snapshot are listed separately in `manifests/vendor_patches.json`.
Do not replace this fork with upstream latest and assume training equivalence.

The following upstream declarations were checked on 2026-11-03 against the
versions referenced by this release. Links identify the material checked.

| Component | License and source | Distribution in VidUEU |
| --- | --- | --- |
| BAGEL code | Apache-2.0; [upstream license](https://github.com/ByteDance-Seed/Bagel/blob/main/LICENSE), preserved in vendor/bagel/LICENSE | Customized code included; upstream notices and source/patch manifests retained |
| BAGEL-7B-MoT weights | Apache-2.0; [pinned model card](https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT/blob/5019f57d168e5816e8f3f701b17cc816bb7cf24b/README.md) | Downloaded separately |
| Ultralytics YOLO code and YOLOv8 weights | AGPL-3.0; [pipeline version license](https://github.com/ultralytics/ultralytics/blob/v8.4.37/LICENSE), [licensing options](https://www.ultralytics.com/license) | pip dependency and separately downloaded weights; not relicensed by Apache-2.0 |
| OmniParser repository code | CC-BY-4.0; [pinned LICENSE](https://github.com/microsoft/OmniParser/blob/b0d5c9f5701f7e2be4771872e6e928da77759df3/LICENSE) | Downloaded separately; compatibility adapter adapts Microsoft OmniParser utilities in memory; upstream attribution and license remain |
| OmniParser icon_detect | AGPL-3.0; [pinned model card, per-folder licenses](https://huggingface.co/microsoft/OmniParser-v2.0/blob/f55d0750e5b94db2125ef0b45b0fa4a85ddc59b4/README.md#license) | This is the YOLOv8 detector used by the current manifest |
| OmniParser icon_caption | MIT; same pinned model card | Downloaded separately |

The recorded BAGEL fork commit is retained as historical provenance; its
GitHub raw-license URL currently returns 404. The bundled LICENSE hash matches
SOURCE_MANIFEST.json and the current public upstream license is Apache-2.0.
This check confirms the license text, not availability of that historical fork
commit from the public upstream repository.

The OmniParser model-card header says MIT, but the per-folder declaration
explicitly makes icon_detect AGPL-3.0. The alternative MIT icon_detect_v3 is
not used by this release. Do not claim that the default tool stack is wholly
permissive. Apache-2.0 covers our original files, not permission to bypass
AGPL obligations for a combined deployment or network service. Retain the
applicable notices/licenses and provide corresponding source as required by
the third-party licenses, or obtain the relevant upstream commercial rights.

Model weights, external code and video/benchmark datasets retain their own
licenses and use restrictions. `manifests/models.yaml` links other assets to
their origins. Downloads are not included in this source tree. The downloader
does not grant rights to model weights, benchmark prompts or source media.
The VidUEU100k dataset is distributed separately at
https://huggingface.co/datasets/Lruca/VidUEU100k under its dataset-card terms.

The public source candidate excludes private credentials, machine-local
deployment configuration, independent post-generation review workflows,
experiment outputs, media and weights. The original workspaces remain intact.
