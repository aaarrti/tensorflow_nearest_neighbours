"""cc_toolchain_config rule for configuring CUDA toolchains on Linux"""

load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "action_config",
    "env_entry",
    "env_set",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
    "tool",
    "tool_path",
    "variable_with_value",
)
load(
    "@bazel_tools//tools/build_defs/cc:action_names.bzl",
    "ASSEMBLE_ACTION_NAME",
    "CC_FLAGS_MAKE_VARIABLE_ACTION_NAME",
    "CLIF_MATCH_ACTION_NAME",
    "CPP_COMPILE_ACTION_NAME",
    "CPP_HEADER_PARSING_ACTION_NAME",
    "CPP_LINK_DYNAMIC_LIBRARY_ACTION_NAME",
    "CPP_LINK_EXECUTABLE_ACTION_NAME",
    "CPP_LINK_NODEPS_DYNAMIC_LIBRARY_ACTION_NAME",
    "CPP_LINK_STATIC_LIBRARY_ACTION_NAME",
    "CPP_MODULE_CODEGEN_ACTION_NAME",
    "CPP_MODULE_COMPILE_ACTION_NAME",
    "C_COMPILE_ACTION_NAME",
    "LINKSTAMP_COMPILE_ACTION_NAME",
    "LTO_BACKEND_ACTION_NAME",
    "LTO_INDEXING_ACTION_NAME",
    "PREPROCESS_ASSEMBLE_ACTION_NAME",
    "STRIP_ACTION_NAME",
)

ACTION_NAMES = struct(
    c_compile = C_COMPILE_ACTION_NAME,
    cpp_compile = CPP_COMPILE_ACTION_NAME,
    linkstamp_compile = LINKSTAMP_COMPILE_ACTION_NAME,
    cc_flags_make_variable = CC_FLAGS_MAKE_VARIABLE_ACTION_NAME,
    cpp_module_codegen = CPP_MODULE_CODEGEN_ACTION_NAME,
    cpp_header_parsing = CPP_HEADER_PARSING_ACTION_NAME,
    cpp_module_compile = CPP_MODULE_COMPILE_ACTION_NAME,
    assemble = ASSEMBLE_ACTION_NAME,
    preprocess_assemble = PREPROCESS_ASSEMBLE_ACTION_NAME,
    lto_indexing = LTO_INDEXING_ACTION_NAME,
    lto_backend = LTO_BACKEND_ACTION_NAME,
    cpp_link_executable = CPP_LINK_EXECUTABLE_ACTION_NAME,
    cpp_link_dynamic_library = CPP_LINK_DYNAMIC_LIBRARY_ACTION_NAME,
    cpp_link_nodeps_dynamic_library = CPP_LINK_NODEPS_DYNAMIC_LIBRARY_ACTION_NAME,
    cpp_link_static_library = CPP_LINK_STATIC_LIBRARY_ACTION_NAME,
    strip = STRIP_ACTION_NAME,
    clif_match = CLIF_MATCH_ACTION_NAME,
    ld_embed_data = "ld_embed_data",
)

def _impl(ctx):
    toolchain_identifier = "local_linux"
    host_system_name = "local"
    target_system_name = "local"
    target_cpu = "local"
    target_libc = "local"
    compiler = "compiler"
    abi_version = "local"
    abi_libc_version = "local"
    cc_target_os = None
    builtin_sysroot = ctx.attr.builtin_sysroot

    all_link_actions = [
        ACTION_NAMES.cpp_link_executable,
        ACTION_NAMES.cpp_link_dynamic_library,
        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
    ]

    action_configs = []

    pic_feature = feature(
        name = "pic",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(flags = ["-fPIC"], expand_if_available = "pic"),
                    flag_group(
                        flags = ["-fPIE"],
                        expand_if_not_available = "pic",
                    ),
                ],
            ),
        ],
    )

    preprocessor_defines_feature = feature(
        name = "preprocessor_defines",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["/D%{preprocessor_defines}"],
                        iterate_over = "preprocessor_defines",
                    ),
                ],
            ),
        ],
    )

    generate_pdb_file_feature = feature(
        name = "generate_pdb_file",
        requires = [
            feature_set(features = ["dbg"]),
            feature_set(features = ["fastbuild"]),
        ],
    )

    linkstamps_feature = feature(
        name = "linkstamps",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["%{linkstamp_paths}"],
                        iterate_over = "linkstamp_paths",
                        expand_if_available = "linkstamp_paths",
                    ),
                ],
            ),
        ],
    )

    unfiltered_compile_flags_feature = feature(
        name = "unfiltered_compile_flags",
        flag_sets = ([
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                ],
                flag_groups = [
                    flag_group(
                        flags = ctx.attr.host_unfiltered_compile_flags,
                    ),
                ],
            ),
        ] if ctx.attr.host_unfiltered_compile_flags else []),
    )

    determinism_feature = feature(
        name = "determinism",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-Wno-builtin-macro-redefined",
                            "-D__DATE__=\"redacted\"",
                            "-D__TIMESTAMP__=\"redacted\"",
                            "-D__TIME__=\"redacted\"",
                        ],
                    ),
                ],
            ),
        ],
    )

    nologo_feature = feature(
        name = "nologo",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_static_library,
                ],
                flag_groups = [flag_group(flags = ["/nologo"])],
            ),
        ],
    )

    supports_pic_feature = feature(name = "supports_pic", enabled = True)

    output_execpath_flags_feature = feature(
        name = "output_execpath_flags",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["/OUT:%{output_execpath}"],
                        expand_if_available = "output_execpath",
                    ),
                ],
            ),
        ],
    )

    default_link_flags_feature = feature(
        name = "default_link_flags",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["/MACHINE:X64"])],
            ),
        ],
    )

    hardening_feature = feature(
        name = "hardening",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-U_FORTIFY_SOURCE",
                            "-D_FORTIFY_SOURCE=1",
                            "-fstack-protector",
                        ],
                    ),
                ],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(flags = ["-Wl,-z,relro,-z,now"])],
            ),
            flag_set(
                actions = [ACTION_NAMES.cpp_link_executable],
                flag_groups = [flag_group(flags = ["-pie", "-Wl,-z,relro,-z,now"])],
            ),
        ],
    )

    supports_dynamic_linker_feature = feature(name = "supports_dynamic_linker", enabled = True)

    warnings_feature = feature(
        name = "warnings",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(
                        flags = ["-Wall"] + ctx.attr.host_compiler_warnings,
                    ),
                ],
            ),
        ],
    )

    compiler_output_flags_feature = feature(
        name = "compiler_output_flags",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.assemble],
                flag_groups = [
                    flag_group(
                        flag_groups = [
                            flag_group(
                                flags = ["/Fo%{output_file}", "/Zi"],
                                expand_if_not_available = "output_preprocess_file",
                            ),
                        ],
                        expand_if_available = "output_file",
                        expand_if_not_available = "output_assembly_file",
                    ),
                ],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                ],
                flag_groups = [
                    flag_group(
                        flag_groups = [
                            flag_group(
                                flags = ["/Fo%{output_file}"],
                                expand_if_not_available = "output_preprocess_file",
                            ),
                        ],
                        expand_if_available = "output_file",
                        expand_if_not_available = "output_assembly_file",
                    ),
                    flag_group(
                        flag_groups = [
                            flag_group(
                                flags = ["/Fa%{output_file}"],
                                expand_if_available = "output_assembly_file",
                            ),
                        ],
                        expand_if_available = "output_file",
                    ),
                    flag_group(
                        flag_groups = [
                            flag_group(
                                flags = ["/P", "/Fi%{output_file}"],
                                expand_if_available = "output_preprocess_file",
                            ),
                        ],
                        expand_if_available = "output_file",
                    ),
                ],
            ),
        ],
    )

    dbg_feature = feature(
        name = "dbg",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-g"])],
            ),
        ],
        implies = ["common"],
    )

    undefined_dynamic_feature = feature(
        name = "undefined-dynamic",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                ],
                flag_groups = [flag_group(flags = ["-undefined", "dynamic_lookup"])],
            ),
        ],
    )

    parse_showincludes_feature = feature(
        name = "parse_showincludes",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_header_parsing,
                ],
                flag_groups = [flag_group(flags = ["/showIncludes"])],
            ),
        ],
    )

    linker_param_file_feature = feature(
        name = "linker_param_file",
        flag_sets = [
            flag_set(
                actions = all_link_actions +
                          [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = ["@%{linker_param_file}"],
                        expand_if_available = "linker_param_file",
                    ),
                ],
            ),
        ],
    )

    supports_interface_shared_libraries_feature = feature(
        name = "supports_interface_shared_libraries",
        enabled = True,
    )

    disable_assertions_feature = feature(
        name = "disable-assertions",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-DNDEBUG"])],
            ),
        ],
    )

    fastbuild_feature = feature(name = "fastbuild", implies = ["common"])

    user_compile_flags_feature = feature(
        name = "user_compile_flags",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["%{user_compile_flags}"],
                        iterate_over = "user_compile_flags",
                        expand_if_available = "user_compile_flags",
                    ),
                ],
            ),
        ],
    )

    compiler_input_flags_feature = feature(
        name = "compiler_input_flags",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["/c", "%{source_file}"],
                        expand_if_available = "source_file",
                    ),
                ],
            ),
        ],
    )

    no_legacy_features_feature = feature(name = "no_legacy_features")

    archiver_flags_feature = feature(
        name = "archiver_flags",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        flags = ["/OUT:%{output_execpath}"],
                        expand_if_available = "output_execpath",
                    ),
                ],
            ),
        ],
    )

    linker_bin_path_feature = feature(
        name = "linker-bin-path",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["-B" + ctx.attr.linker_bin_path])],
            ),
        ],
    )

    opt_feature = feature(
        name = "opt",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [
                    flag_group(
                        flags = ["-g0", "-O2", "-ffunction-sections", "-fdata-sections"],
                    ),
                ],
            ),
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                ],
                flag_groups = [flag_group(flags = ["-Wl,--gc-sections"])],
            ),
        ],
        implies = ["common", "disable-assertions"],
    )

    include_paths_feature = feature(
        name = "include_paths",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["/I%{quote_include_paths}"],
                        iterate_over = "quote_include_paths",
                    ),
                    flag_group(
                        flags = ["/I%{include_paths}"],
                        iterate_over = "include_paths",
                    ),
                    flag_group(
                        flags = ["/I%{system_include_paths}"],
                        iterate_over = "system_include_paths",
                    ),
                ],
            ),
        ],
    )

    shared_flag_feature = feature(
        name = "shared_flag",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [flag_group(flags = ["/DLL"])],
            ),
        ],
    )

    frame_pointer_feature = feature(
        name = "frame-pointer",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.c_compile, ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-fno-omit-frame-pointer"])],
            ),
        ],
    )

    build_id_feature = feature(
        name = "build-id",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["-Wl,--build-id=md5", "-Wl,--hash-style=gnu"],
                    ),
                ],
            ),
        ],
    )

    sysroot_feature = feature(
        name = "sysroot",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["--sysroot=%{sysroot}"],
                        iterate_over = "sysroot",
                        expand_if_available = "sysroot",
                    ),
                ],
            ),
        ],
    )

    cuda_path_feature = feature(
        name = "cuda_path",
        enabled = True,
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.assemble,
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    ACTION_NAMES.cpp_module_compile,
                    ACTION_NAMES.cpp_module_codegen,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["--cuda-path=" + ctx.attr.cuda_path],
                    ),
                ],
            ),
        ],
    )

    def_file_feature = feature(
        name = "def_file",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["/DEF:%{def_file_path}", "/ignore:4070"],
                        expand_if_available = "def_file_path",
                    ),
                ],
            ),
        ],
    )

    stdlib_feature = feature(
        name = "stdlib",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [flag_group(flags = ["-lstdc++"])],
            ),
        ],
    )

    no_stripping_feature = feature(name = "no_stripping")

    alwayslink_feature = feature(
        name = "alwayslink",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                ],
                flag_groups = [flag_group(flags = ["-Wl,-no-as-needed"])],
            ),
        ],
    )

    input_param_flags_feature = feature(
        name = "input_param_flags",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [
                    flag_group(
                        flags = ["/IMPLIB:%{interface_library_output_path}"],
                        expand_if_available = "interface_library_output_path",
                    ),
                ],
            ),
            flag_set(
                actions = all_link_actions +
                          [ACTION_NAMES.cpp_link_static_library],
                flag_groups = [
                    flag_group(
                        iterate_over = "libraries_to_link",
                        flag_groups = [
                            flag_group(
                                iterate_over = "libraries_to_link.object_files",
                                flag_groups = [flag_group(flags = ["%{libraries_to_link.object_files}"])],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file_group",
                                ),
                            ),
                            flag_group(
                                flag_groups = [flag_group(flags = ["%{libraries_to_link.name}"])],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "object_file",
                                ),
                            ),
                            flag_group(
                                flag_groups = [flag_group(flags = ["%{libraries_to_link.name}"])],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "interface_library",
                                ),
                            ),
                            flag_group(
                                flag_groups = [
                                    flag_group(
                                        flags = ["%{libraries_to_link.name}"],
                                        expand_if_false = "libraries_to_link.is_whole_archive",
                                    ),
                                    flag_group(
                                        flags = ["/WHOLEARCHIVE:%{libraries_to_link.name}"],
                                        expand_if_true = "libraries_to_link.is_whole_archive",
                                    ),
                                ],
                                expand_if_equal = variable_with_value(
                                    name = "libraries_to_link.type",
                                    value = "static_library",
                                ),
                            ),
                        ],
                        expand_if_available = "libraries_to_link",
                    ),
                ],
            ),
        ],
    )

    no_canonical_prefixes_feature = feature(
        name = "no-canonical-prefixes",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_link_executable,
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
                ],
                flag_groups = [
                    flag_group(
                        flags = [
                            "-no-canonical-prefixes",
                        ] + ctx.attr.extra_no_canonical_prefixes_flags,
                    ),
                ],
            ),
        ],
    )

    has_configured_linker_path_feature = feature(name = "has_configured_linker_path")
    copy_dynamic_libraries_to_binary_feature = feature(name = "copy_dynamic_libraries_to_binary")

    user_link_flags_feature = feature(
        name = "user_link_flags",
        flag_sets = [
            flag_set(
                actions = all_link_actions,
                flag_groups = [
                    flag_group(
                        flags = ["%{user_link_flags}"],
                        iterate_over = "user_link_flags",
                        expand_if_available = "user_link_flags",
                    ),
                ],
            ),
        ],
    )

    cpp17_feature = feature(
        name = "c++17",
        flag_sets = [
            flag_set(
                actions = [ACTION_NAMES.cpp_compile],
                flag_groups = [flag_group(flags = ["-std=c++17"])],
            ),
        ],
    )

    common_feature = feature(
        name = "common",
        implies = [
            "stdlib",
            "c++17",
            "determinism",
            "alwayslink",
            "hardening",
            "warnings",
            "frame-pointer",
            "build-id",
            "no-canonical-prefixes",
            "linker-bin-path",
        ],
    )

    features = [
        cpp17_feature,
        stdlib_feature,
        determinism_feature,
        alwayslink_feature,
        pic_feature,
        hardening_feature,
        warnings_feature,
        frame_pointer_feature,
        build_id_feature,
        no_canonical_prefixes_feature,
        disable_assertions_feature,
        linker_bin_path_feature,
        common_feature,
        opt_feature,
        fastbuild_feature,
        dbg_feature,
        supports_dynamic_linker_feature,
        supports_pic_feature,
    ]
    if ctx.attr.cuda_path:
        features += [cuda_path_feature]

    cxx_builtin_include_directories = ctx.attr.builtin_include_directories

    tool_paths = [
        tool_path(name = "gcc", path = ctx.attr.host_compiler_path),
        tool_path(name = "ar", path = ctx.attr.host_compiler_prefix + "/ar"),
        tool_path(name = "compat-ld", path = ctx.attr.host_compiler_prefix + "/ld"),
        tool_path(name = "cpp", path = ctx.attr.host_compiler_prefix + "/cpp"),
        tool_path(name = "dwp", path = ctx.attr.host_compiler_prefix + "/dwp"),
        tool_path(name = "gcov", path = ctx.attr.host_compiler_prefix + "/gcov"),
        tool_path(name = "ld", path = ctx.attr.host_compiler_prefix + "/ld"),
        tool_path(name = "nm", path = ctx.attr.host_compiler_prefix + "/nm"),
        tool_path(name = "objcopy", path = ctx.attr.host_compiler_prefix + "/objcopy"),
        tool_path(name = "objdump", path = ctx.attr.host_compiler_prefix + "/objdump"),
        tool_path(name = "strip", path = ctx.attr.host_compiler_prefix + "/strip"),
    ]

    out = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(out, "Fake executable")
    return [
        cc_common.create_cc_toolchain_config_info(
            ctx = ctx,
            features = features,
            action_configs = action_configs,
            artifact_name_patterns = [],
            cxx_builtin_include_directories = cxx_builtin_include_directories,
            toolchain_identifier = toolchain_identifier,
            host_system_name = host_system_name,
            target_system_name = target_system_name,
            target_cpu = target_cpu,
            target_libc = target_libc,
            compiler = compiler,
            abi_version = abi_version,
            abi_libc_version = abi_libc_version,
            tool_paths = tool_paths,
            make_variables = [],
            builtin_sysroot = builtin_sysroot,
            cc_target_os = cc_target_os,
        ),
        DefaultInfo(
            executable = out,
        ),
    ]

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {
        "cpu": attr.string(mandatory = True, values = ["local"]),
        "builtin_include_directories": attr.string_list(),
        "extra_no_canonical_prefixes_flags": attr.string_list(),
        "host_compiler_path": attr.string(),
        "host_compiler_prefix": attr.string(),
        "host_compiler_warnings": attr.string_list(),
        "host_unfiltered_compile_flags": attr.string_list(),
        "linker_bin_path": attr.string(),
        "builtin_sysroot": attr.string(),
        "cuda_path": attr.string(),
    },
    provides = [CcToolchainConfigInfo],
    executable = True,
)
