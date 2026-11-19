#!/usr/bin/env bash
# Run a command; if it fails, post its last lines as a note on the run page,
# so the error can be read without opening the full log.
set -o pipefail
"$@" 2>&1 | tee step.log
status=$?
if [ $status -ne 0 ]; then
  echo "::error title=$* failed::$(tail -n 25 step.log | tr '\n' '|' | tail -c 3000)"
fi
exit $status
