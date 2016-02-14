# Copyright (c) 2016 Dominik Zeromski <dzeromsk@gmail.com>

CFLAGS+=$(shell pkg-config --cflags libdrm_intel)
LDLIBS+=$(shell pkg-config --libs libdrm_intel)

all: example_bdw example_hsw example_skl
clean:
	rm -f example_bdw example_hsw example_skl
