using Pkg

Pkg.develop(PackageSpec(path=pwd()))
Pkg.instantiate()

using RadialCalderon
using Documenter
using DocumenterCitations
using Literate

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:authoryear)

open(joinpath(joinpath(@__DIR__, "src"), "index.md"), "w") do io
    println(
        io,
        """
        ```@meta
        EditURL = "https://github.com/rpetit/RadialCalderon.jl/blob/main/README.md"
        ```
        """,
    )
    for line in eachline(joinpath(dirname(@__DIR__), "README.md"))
        println(io, line)
    end
end

examples_jl_path = joinpath(dirname(@__DIR__), "examples")
examples_md_path = joinpath(@__DIR__, "src", "examples")

for file in readdir(examples_md_path)
    if endswith(file, ".md")
        rm(joinpath(examples_md_path, file))
    end
end

for file in readdir(examples_jl_path)
    Literate.markdown(joinpath(examples_jl_path, file), examples_md_path)
end

function literate_title(path)
    l = first(readlines(path))
    return l[3:end]
end

pages = [
    "Home" => "index.md",
    "Tutorials" => [
        "Forward map" => joinpath("examples", "forward.md")
    ],
    "API reference" => "api.md"
]

makedocs(;sitename="RadialCalderon.jl", pages=pages, plugins=[bib])

Pkg.rm("RadialCalderon")