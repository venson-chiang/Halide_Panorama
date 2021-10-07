include Makefile.inc

.PHONY: build clean test

build: $(BIN)/${HL_TARGET}/output.png

export LD_LIBRARY_PATH=/mnt/d/Software/Halide-12/distrib/lib

$(GENERATOR_BIN)/main: main.cpp src/utils.cpp src/panorama.cpp
	@mkdir -p $(@D)
	g++ -I /mnt/d/Software/Halide-12/distrib/include -I /mnt/d/Software/Halide-12/distrib/tools -I ./ \
	-L /mnt/d/Software/Halide-12/distrib/lib $^ -o $@ -rdynamic -lHalide -lpthread -ldl -ljpeg -lpng -larmadillo -std=c++11

$(BIN)/%/output.png: $(GENERATOR_BIN)/main 
	@mkdir -p $(@D)
	$< images iphone-1.png iphone-2.png
	
#iphone-1.png iphone-2.png
#guedelon-1.png guedelon-2.png guedelon-3.png guedelon-4.png
#boston1-1.png boston1-2.png boston1-3.png boston1-4.png boston1-5.png
#convention-1.png convention-2.png convention-3.png convention-4.png 
#vancouver-6.png vancouver-5.png vancouver-4.png vancouver-3.png vancouver-2.png
#sunset-1.png sunset-2.png

clean:
	rm -rf $(BIN)

test: $(BIN)/${HL_TARGET}/output.png
	 