set(TARGET_NAME monteCarlo)

set(${TARGET_NAME}_SOURCES
    main.cpp
    CameraController.cpp		
)


add_executable(${TARGET_NAME} ${${TARGET_NAME}_SOURCES})
target_link_libraries(${TARGET_NAME} PUBLIC tga_vulkan tga_utils ${CMAKE_THREAD_LIBS_INIT})
add_dependencies(${TARGET_NAME} tga_shaders)
if(WIN32)
    set_property(TARGET ${TARGET_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif(WIN32)

