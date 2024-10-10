#!/bin/bash

child_pid=0

cleanup () {
  echo tgi interrupted, cleaning
  sig=$(($? - 128))
  if [ $sig -le 0 ];then
    sig=TERM
  fi
  echo signal $sig received
  if [ $child_pid -gt 0 ] && ps -p $child_pid > /dev/null;then
    echo kill child $child_pid
    kill -$sig $child_pid
    wait $child_pid
    child_pid=0
  fi
}

trap cleanup INT TERM HUP EXIT ERR

ldconfig 2>/dev/null || echo 'unable to refresh ld cache, not a big deal in most cases'

text-generation-launcher $@ &

child_pid=$!

wait $child_pid
