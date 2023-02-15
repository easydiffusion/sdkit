from sdkit.models import scan_model
from sdkit.utils import log

# note: models are scanned automatically when load_model() is called
# this example shows you how to scan models, for e.g. if you want to scan
# lots of models, and then disable model-scanning each time a model is loaded
# to improve load time.

model_path = "D:\\path\\to\\malicious_model.ckpt"
scan_result = scan_model(model_path)

if scan_result.issues_count > 0 or scan_result.infected_files > 0:
    log.warn(
        ":warning: [bold red]Scan %s: %d scanned, %d issue, %d infected.[/bold red]"
        % (model_path, scan_result.scanned_files, scan_result.issues_count, scan_result.infected_files)
    )
else:
    log.debug(
        "Scan %s: [green]%d scanned, %d issue, %d infected.[/green]"
        % (model_path, scan_result.scanned_files, scan_result.issues_count, scan_result.infected_files)
    )
