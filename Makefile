COMPILER=gcc
OUTPUT=cv
INPUT=refactortest.c
CUDALIBS = -lnppc -lnppi -lcublas -lcublas -lcusolver -lcudart
GTKCFLAGS = -IC:/gtk/include/gtk-2.0 -IC:/gtk/lib/gtk-2.0/include -IC:/gtk/include/pango-1.0 -IC:/gtk/include/gio-unix-2.0/ -IC:/gtk/include/cairo -IC:/gtk/include/atk-1.0 -IC:/gtk/include/cairo -IC:/gtk/include/pixman-1 -IC:/gtk/include/gdk-pixbuf-2.0 -IC:/gtk/include/libpng16 -IC:/gtk/include/pango-1.0 -IC:/gtk/include/harfbuzz -IC:/gtk/include/pango-1.0 -IC:/gtk/include/glib-2.0 -IC:/gtk/lib/glib-2.0/include -IC:/gtk/include/freetype2 -IC:/gtk/include/libpng16 -IC:/gtk/include/freetype2 -IC:/gtk/include/libpng16
GTKLIBS = -lgtk-win32-2.0 -lgdk-win32-2.0 -lpangocairo-1.0 -lgio-2.0 -latk-1.0 -lcairo -lgdk_pixbuf-2.0 -lgio-2.0 -lpangoft2-1.0 -lpango-1.0 -lgobject-2.0 -lglib-2.0 -lintl -lfontconfig -lfreetype
GTKPATH = C:/gtk

CV_MATRIXUTIL = utils/CUDAMatrixUtil.cu
CV_IMAGEUTIL = utils/CUDAImageUtil.cu

CUPATH = C:/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/CUDA/v8.0

.PHONY : utils app all

utils:
	nvcc -shared --compiler-options="-D EXPORTING" -I./ -I./bin -o CUDAMatrixUtil $(CV_MATRIXUTIL) $(CUDALIBS)
	nvcc -shared --compiler-options="-D EXPORTING" -I./ -I./bin -o CUDAImageUtil $(CV_IMAGEUTIL) -L./bin -lCUDAMatrixUtil $(CUDALIBS)

app:
	nvcc -I./ -I./bin $(GTKCFLAGS) -o $(OUTPUT) $(INPUT) -L./ -L$(GTKPATH)/lib -lCUDAMatrixUtil -lCUDAImageUtil $(GTKLIBS)

all: utils app

clean:
	rm -rf *.obj
	rm -rf *.a
	rm -rf *.exe
	rm -rf *.lib
	rm -rf *.exp
	rm -rf bin/*.a
	rm -rf bin/*.exe
	rm -rf bin/*.lib
	rm -rf bin/*.exp
	rm -rf bin/*.obj
	rm -rf bin/*.dll
