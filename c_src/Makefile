CC=g++
CFLAGS=-O3 -Wall -g -march=native -mavx2 -I eigen-3.4.0/ -I ../assets/ -I json/include -I "./GTnn"

BIN=gtnn_max gtnn_sum gtnn_stream dbgtnn_sum classwise_dbgtnn_sum classwise_gtnn_sum

all: $(BIN)

gtnn_max: example_GTmax.cpp
	$(CC) $(CFLAGS) $^ -o $@

gtnn_sum: example_GTsum.cpp
	$(CC) $(CFLAGS) $^ -o $@

gtnn_stream: example_GTstreaming.cpp
	$(CC) $(CFLAGS) $^ -o $@

dbgtnn_sum: example_DBGTsum.cpp
	$(CC) $(CFLAGS) $^ -o $@

classwise_dbgtnn_sum: classwise_DBGTsum.cpp
	$(CC) $(CFLAGS) $^ -o $@

classwise_gtnn_sum: classwise_GTsum.cpp
	$(CC) $(CFLAGS) $^ -o $@

clean: 
	rm -rf $(BIN) *.dSYM
