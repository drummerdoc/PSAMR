target_sources(${PSAMR_exe_name}
   PRIVATE
     PSAMR_defaults.H
     PSAMR_params.H
     PSAMR_queries.H
   )

target_include_directories(${PSAMR_exe_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
