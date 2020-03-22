#!/usr/bin/env bash
python3 -m api.app &
cd gui &&
npm run dev