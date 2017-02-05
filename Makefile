COMPILER=gcc
OUTPUT=cv
INPUT=refactortest.c
CUDALIBS = -lnppc -lnppi -lcublas -lcublas -lcusolver -lcudart
GTKCFLAGS = -IC:/gtk/include/gtk-2.0 -IC:/gtk/lib/gtk-2.0/include -IC:/gtk/include/pango-1.0 -IC:/gtk/include/gio-unix-2.0/ -IC:/gtk/include/cairo -IC:/gtk/include/atk-1.0 -IC:/gtk/include/cairo -IC:/gtk/include/pixman-1 -IC:/gtk/include/gdk-pixbuf-2.0 -IC:/gtk/include/libpng16 -IC:/gtk/include/pango-1.0 -IC:/gtk/include/harfbuzz -IC:/gtk/include/pango-1.0 -IC:/gtk/include/glib-2.0 -IC:/gtk/lib/glib-2.0/include -IC:/gtk/include/freetype2 -IC:/gtk/include/libpng16 -IC:/gtk/include/freetype2 -IC:/gtk/include/libpng16
GTKLIBS = -lgtk-win32-2.0 -lgdk-win32-2.0 -lpangocairo-1.0 -lgio-2.0 -latk-1.0 -lcairo -lgdk_pixbuf-2.0 -lgio-2.0 -lpangoft2-1.0 -lpango-1.0 -lgobject-2.0 -lglib-2.0 -lintl -lfontconfig -lfreetype
GTKPATH = C:/gtk

CV_MATRIXUTIL = utils/CUDAMatrixUtil.cu
CV_IMAGEUTIL = utils/CUDAImageUtil.cu

CV_UTILS = IOUtil DrawUtil
CV_STRUCTS = ListNode BinaryTreeNode HeapNode List Keypoint Heap PriorityQ
CV_OPERATORS = Extractor
CV_GENERATORS = Filters

CUPATH = C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v8.0

.PHONY : structs utils operators generators app all

all: clean structs utils generators operators

_structs:
	nvcc -shared --compiler-options="-D EXPORTING" -I./ -o lib/$(REC_STRUCT) ./structs/$(REC_STRUCT).c -L./lib $(foreach lib,$(wildcard ./lib/*.lib),-l$(subst ./lib/,,$(subst .lib,,$(lib))))

structs:
	echo "BUILDING STRUCTS\n"
	$(foreach obj,$(CV_STRUCTS),$(MAKE) REC_STRUCT="$(obj)" _structs;)

_utils:
	nvcc -shared --compiler-options="-D EXPORTING" -I./ $(GTKCFLAGS) -o lib/$(REC_UTIL) ./utils/$(REC_UTIL).c -L./lib -L$(GTKPATH)/lib $(GTKLIBS) $(foreach lib,$(wildcard ./lib/*.lib),-l$(subst ./lib/,,$(subst .lib,,$(lib))))

utils:
	echo "BUILDING UTILS\n"
	nvcc -shared --compiler-options="-D EXPORTING" -I./ -I./lib -o lib/CUDAMatrixUtil $(CV_MATRIXUTIL) -L./lib $(CUDALIBS)
	nvcc -shared --compiler-options="-D EXPORTING" -I./ -I./lib -o lib/CUDAImageUtil $(CV_IMAGEUTIL) -L./lib -lCUDAMatrixUtil $(CUDALIBS)
	$(foreach obj,$(CV_UTILS),$(MAKE) REC_UTIL="$(obj)" _utils;)

_operators:
	nvcc -shared --compiler-options="-D EXPORTING" -I./ -o lib/$(REC_OPERATOR) ./operators/$(REC_OPERATOR).c -L./lib $(foreach lib,$(wildcard ./lib/*.lib),-l$(subst ./lib/,,$(subst .lib,,$(lib))))

operators:
	echo "BUILDING OPERATORS\n"
	$(foreach obj,$(CV_OPERATORS),$(MAKE) REC_OPERATOR="$(obj)" _operators;)

_generators:
	nvcc -shared --compiler-options="-D EXPORTING" -I./ -o lib/$(REC_GENERATOR) ./generators/$(REC_GENERATOR).c -L./lib $(foreach lib,$(wildcard ./lib/*.lib),-l$(subst ./lib/,,$(subst .lib,,$(lib))))

generators:
	echo "BUILDING GENERATOR\n"
	$(foreach obj,$(CV_GENERATORS),$(MAKE) REC_GENERATOR="$(obj)" _generators;)


app:
	nvcc -I./ -I./lib/ $(GTKCFLAGS) -o $(OUTPUT) $(INPUT) -L./ -L./lib/ -L$(GTKPATH)/lib $(foreach lib,$(wildcard ./lib/*.lib),-l$(subst ./lib/,,$(subst .lib,,$(lib)))) $(GTKLIBS)


clean:
	rm -rf *.obj
	rm -rf *.a
	rm -rf *.exe
	rm -rf *.lib
	rm -rf *.exp
	rm -rf lib/*.a
	rm -rf lib/*.exe
	rm -rf lib/*.lib
	rm -rf lib/*.exp
	rm -rf lib/*.obj
	rm -rf lib/*.dll
