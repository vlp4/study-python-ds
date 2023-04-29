#!/bin/sh

wget -O chess.zip "https://storage.yandexcloud.net/study-files/chess.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=NTOqBctB4eJ5kdVuQXzn%2F20230429%2Fru-central1%2Fs3%2Faws4_request&X-Amz-Date=20230429T081435Z&X-Amz-Expires=2592000&X-Amz-Signature=DF148079984D61AECADEEC4BE3E2C2D94E5E71E952B6D8BB83C2E676E9167C55&X-Amz-SignedHeaders=host"
mkdir -p data data/chess
unzip -d data/chess_source -o chess.zip