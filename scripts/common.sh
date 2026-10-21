#!/bin/bash

timestamp() {

    date '+%Y-%m-%d %H:%M:%S'
}

log_info() {

    echo ""
    echo "[$(timestamp)] INFO  $*"
}

log_warn() {

    echo ""
    echo "[$(timestamp)] WARN  $*"
}

log_error() {

    echo ""
    echo "[$(timestamp)] ERROR $*"
}