COMPILER=g++
OUTPUT=cv
INPUT=refactortest.c
#GTKCFLAGS = -I/usr/include/gtk-3.0 -I/usr/include/at-spi2-atk/2.0 -I/usr/include/at-spi-2.0 -I/usr/include/dbus-1.0 -I/usr/lib/dbus-1.0/include -I/usr/include/gtk-3.0 -I/usr/include/gio-unix-2.0/ -I/usr/include/cairo -I/usr/include/pango-1.0 -I/usr/include/harfbuzz -I/usr/include/pango-1.0 -I/usr/include/atk-1.0 -I/usr/include/cairo -I/usr/include/pixman-1 -I/usr/include/freetype2 -I/usr/include/libpng16 -I/usr/include/freetype2 -I/usr/include/libpng16 -I/usr/include/gdk-pixbuf-2.0 -I/usr/include/libpng16 -I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include
#GTKLIBS = -lgtk-3 -lgdk-3 -lpangocairo-1.0 -lpango-1.0 -latk-1.0 -lcairo-gobject -lcairo -lgdk_pixbuf-2.0 -lgio-2.0 -lgobject-2.0 -lglib-2.0 -lintl
CUDALIBS = -lnppc -lnppi -lcublas -lcublas -lcusolver -lcudart
GTKCFLAGS = -I/usr/include/gtk-3.0 -I/usr/include/at-spi2-atk/2.0 -I/usr/include/at-spi-2.0 -I/usr/include/dbus-1.0 -I/usr/lib/dbus-1.0/include -I/usr/include/gtk-3.0 -I/usr/include/gio-unix-2.0/ -I/usr/include/cairo -I/usr/include/pango-1.0 -I/usr/include/harfbuzz -I/usr/include/pango-1.0 -I/usr/include/atk-1.0 -I/usr/include/cairo -I/usr/include/pixman-1 -I/usr/include/freetype2 -I/usr/include/libpng16 -I/usr/include/freetype2 -I/usr/include/libpng16 -I/usr/include/gdk-pixbuf-2.0 -I/usr/include/libpng16 -I/usr/include/glib-2.0 -I/usr/lib/glib-2.0/include
GTKLIBS = -lgdk_pixbuf-2.0 -lgio-2.0 -lgobject-2.0 -lglib-2.0 -lintl

CV_NATIVEDIRS = utils/CUDAMatrixUtil.cu utils/CUDAImageUtil.cu

build:
	$(foreach dir,$(CV_NATIVEDIRS),nvcc -o $(subst .cu,,$(dir)) --shared $(dir) $(CUDALIBS);)
	$(COMPILER) -I./utils -o $(OUTPUT) $(INPUT) -L./utils $(foreach dir,$(CV_NATIVEDIRS),-l$(subst utils/,,$(subst .cu,,$(dir))))

clean:
	rm -rf *.obj
	rm -rf *.a
	rm -rf *.exe
	rm -rf *.lib
	rm -rf *.exp
	rm -rf utils/*.a
	rm -rf utils/*.exe
	rm -rf utils/*.lib
	rm -rf utils/*.exp
	rm -rf utils/*.obj
	rm -rf utils/*.dll
