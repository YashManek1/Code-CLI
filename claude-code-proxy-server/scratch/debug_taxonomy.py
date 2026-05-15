import sys
import os
sys.path.append(os.getcwd())

from core.healing.taxonomy import FailureTaxonomy, FailureType
error = TimeoutError("request timed out")
error_msg = str(error).lower()
print(f"error_msg: '{error_msg}'")
print(f"'timeout' in error_msg: {'timeout' in error_msg}")

fail = FailureTaxonomy.classify(error)
print(f"Type: {fail.type}")
