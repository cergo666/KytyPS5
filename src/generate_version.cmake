set(KYTY_GIT_VERSION "unknown")
set(KYTY_GIT_HASH "unknown")
set(KYTY_GIT_REVISION "unknown")
if(GIT_EXECUTABLE)
	execute_process(
		COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty
		WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
		OUTPUT_VARIABLE KYTY_GIT_VERSION
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE GIT_RESULT
		ERROR_QUIET
	)
	if(NOT GIT_RESULT EQUAL 0)
		set(KYTY_GIT_VERSION "unknown")
	endif()

	execute_process(
		COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
		WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
		OUTPUT_VARIABLE KYTY_GIT_REVISION
		OUTPUT_STRIP_TRAILING_WHITESPACE
		RESULT_VARIABLE GIT_HASH_RESULT
		ERROR_QUIET
	)
	if(NOT GIT_HASH_RESULT EQUAL 0)
		set(KYTY_GIT_HASH "unknown")
		set(KYTY_GIT_REVISION "unknown")
	else()
		string(SUBSTRING "${KYTY_GIT_REVISION}" 0 7 KYTY_GIT_HASH)
		execute_process(
			COMMAND "${GIT_EXECUTABLE}" diff-index --quiet HEAD --
			WORKING_DIRECTORY "${GIT_WORKING_DIRECTORY}"
			RESULT_VARIABLE GIT_DIRTY_RESULT
			ERROR_QUIET
		)
		if(NOT GIT_DIRTY_RESULT EQUAL 0)
			string(APPEND KYTY_GIT_HASH "-dirty")
		endif()
	endif()
endif()
configure_file("${INPUT_FILE}" "${OUTPUT_FILE}")

# Hash of the sources matching glob patterns relative to SHADER_CACHE_SOURCE_DIR (paths and
# contents) and of the toolchain.
function(kyty_source_hash out)
	set(digest "${SHADER_CACHE_TOOLCHAIN}\n")
	foreach(pattern IN LISTS ARGN)
		file(GLOB_RECURSE sources LIST_DIRECTORIES false RELATIVE "${SHADER_CACHE_SOURCE_DIR}"
			"${SHADER_CACHE_SOURCE_DIR}/${pattern}")
		list(SORT sources)
		foreach(source IN LISTS sources)
			file(SHA256 "${SHADER_CACHE_SOURCE_DIR}/${source}" source_hash)
			string(APPEND digest "${source} ${source_hash}\n")
		endforeach()
	endforeach()
	string(SHA256 hash "${digest}")
	string(SUBSTRING "${hash}" 0 16 hash)
	set(${out} "${hash}" PARENT_SCOPE)
endfunction()

if(SHADER_CACHE_SOURCE_DIR)
	# Records hold program inputs only and are compiled by the running build, so they depend on
	# the code that defines their format and the program key: the shader tree (recompiler, input
	# structures, static keys), the record serializer and the GPU format enums. Other graphics
	# changes keep the records (personal build: upstream hashes the whole graphics tree).
	kyty_source_hash(KYTY_GRAPHICS_CACHE_VERSION "graphics/shader/*"
		"graphics/host_gpu/renderer/pipeline/shaderCacheFile.*" "graphics/guest_gpu/gpu_format*")
	configure_file("${SHADER_CACHE_INPUT_FILE}" "${SHADER_CACHE_OUTPUT_FILE}")
endif()
