def _tpl(repository_ctx, tpl, substitutions = {}):
    repository_ctx.template(
        tpl,
        Label("//metal:%s.tpl" % tpl),
        substitutions,
    )

def _metal_configure_impl(repository_ctx):
    is_metal_configured = repository_ctx.os.environ.get("TF_NEED_METAL") == "1"
    _tpl(repository_ctx, "BUILD", {})
    _tpl(repository_ctx, "build_defs.bzl", {"%{metal_is_configured}": str(is_metal_configured)})

metal_configure = repository_rule(
    implementation = _metal_configure_impl,
    environ = ["TF_NEED_METAL"],
)
