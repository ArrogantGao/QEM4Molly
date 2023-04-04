using QEM4Molly
using Documenter

DocMeta.setdocmeta!(QEM4Molly, :DocTestSetup, :(using QEM4Molly); recursive=true)

makedocs(;
    modules=[QEM4Molly],
    authors="Xuanzhao Gao",
    repo="https://github.com/ArrogantGao/QEM4Molly.jl/blob/{commit}{path}#{line}",
    sitename="QEM4Molly.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ArrogantGao.github.io/QEM4Molly.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ArrogantGao/QEM4Molly.jl",
    devbranch="main",
)
