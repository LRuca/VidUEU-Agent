# Diagnostics

Start with `vidueu doctor --config config.local.yaml --lightweight`. Add
`--check-assets --check-media` after setup/download. `--check-endpoint` queries
the VLM model list. Do not add `--check-gpu` for CPU smoke.

| Code | Action |
| --- | --- |
| unexpanded_environment_variables | Set the named .env variable or explicit input/output override. |
| duplicate_config_key, unknown_config_field | Correct YAML; compare config example. |
| dotenv_reference_cycle | Remove cyclic references; .env is never executed. |
| input_missing, no_video_inputs | Point input at an existing supported video/directory. |
| output_inside_input | Put output outside the source directory. |
| api_credentials_missing | Set the key variable/file; doctor makes no paid request. |
| api_budget_exhausted | Plan a new authorized budget; restarts do not reset the ledger. |
| vlm_served_model_mismatch | Use the model ID advertised by your server. |
| device_outside_visible_mask | Use logical ordinals within selected physical GPUs. |
| asset_file_missing, model_weights_missing | Download/verify selected assets or correct model_root. |
| omni_python_missing | Install its environment and set tool_python, or deselect it. |
| low_disk, dataset_write_failed | Check space/permissions and the owned log. |
| image_missing, image_corrupt | Restore the exact image before export. |
| image_outside_asset_roots | Verify provenance and explicitly authorize the right root. |
| export_source_changed | Stop the producer and retry to a new destination. |
| metadata_snapshot_missing | Supply structured snapshots or a legacy converter. |
| recipe_deployment_incomplete | Read reproduce plan and complete the listed inputs. |

CLI diagnostics withhold provider response bodies and arbitrary exception text.
Owned logs can contain dataset text and local paths; inspect before sharing.
Status is read-only; no diagnostics command performs automatic cleanup/restarts.
