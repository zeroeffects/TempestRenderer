#ADD_DEFINITIONS(-DUSE_SIMD)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

SET(KISSFFT_SRC
	kiss_fft.c
    tools/kiss_fftr.c
	tools/kiss_fftnd.c
	tools/kiss_fftndr.c
	)

SET(KISSFFT_INC
	kiss_fft.h
	tools/kiss_fftr.h
    tools/kiss_fftnd.h
	tools/kiss_fftndr.h
	)

ADD_LIBRARY(kissfft STATIC ${KISSFFT_SRC} ${KISSFFT_INC})