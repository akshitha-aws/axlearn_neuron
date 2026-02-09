#!/bin/bash

set -ex

# Check if any arguments were provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <test_suite1> [test_suite2 ...]"
    echo "Available suites: presubmit, 12b, 50b, 150b, qwen3-30b, switch-base, switch-large, mixtral-50b, llama4-scout, deepseek-v3, qwen3-235b, switch-xxl, llama4-maverick"
    exit 1
fi

# Generate unique ID for this test run
id=$(date +"%Y%m%d_%H%M%S")
# Use Kaizen env vars if set, otherwise generate defaults
TEST_LOGDIR=${KAIZEN_TEST_LOG_DIR:-test_artifacts/$id}
GOLDENS_DIR=${KAIZEN_GOLDENS_DIR:-test_goldens/$id}
JAX_CC_DIR=""

# Test execution function
run_suite() {
    local suite=$1
    echo "Running test suite: $suite"
    mkdir -p $TEST_LOGDIR/${suite}
    
    if [ "$suite" = "150b" ]; then
        ./aws-testing/unit_test.sh 150bdev ${suite} $TEST_LOGDIR $GOLDENS_DIR $JAX_CC_DIR 2>&1 | tee $TEST_LOGDIR/${suite}/150bdev.log
    fi
    ./aws-testing/unit_test.sh integ $suite $TEST_LOGDIR $GOLDENS_DIR $JAX_CC_DIR 2>&1 | tee $TEST_LOGDIR/${suite}/integ.log
}

# Summary function
summary() {
    local suite=$1
    echo "---------------------------------"
    num_passed=$(grep -re 'PASSED' $TEST_LOGDIR/$suite/*.log | wc -l)
    num_failed=$(grep -re 'FAILED' $TEST_LOGDIR/$suite/*.log | wc -l)
    total_num=$(($num_passed + $num_failed))
    echo "Test suite: $suite, Total tests $total_num"
    echo "Number of tests passed: $num_passed"
    echo "Number of tests failed: $num_failed"
    if [ $num_failed -gt 0 ]; then
        echo "Failed tests:"
        grep -hre 'FAILED' $TEST_LOGDIR/$suite/*.log
    fi
}

# Create test directory
mkdir -p $TEST_LOGDIR

# Run specified test suites in parallel
for suite in "$@"; do
    (run_suite $suite && summary $suite)
done

echo "All tests finished"

# Send the data via scuba
python ./aws-testing/batch_parser.py --test_directory="$TEST_LOGDIR" --event_id="$VS_EVENT_ID"
