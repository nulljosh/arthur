CC = cc
CFLAGS = -O2 -Wall -std=c99
TARGET = arthur

$(TARGET): arthur.c
	$(CC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -f $(TARGET)

.PHONY: clean
