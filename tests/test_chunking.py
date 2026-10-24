from surgidoc.retrieval.chunking import MAX_WORDS, chunk_article

META = {"key": "DOC", "title": "T", "url": "u", "license": "cc by"}


def write(tmp_path, body):
    path = tmp_path / "a.xml"
    path.write_text(f"<article><front><abstract><p>Short abstract.</p></abstract></front>"
                    f"<body>{body}</body></article>")
    return path


def test_sections_paths_and_skipped_back_matter(tmp_path):
    words = " ".join(["word"] * 130)
    path = write(tmp_path, f"""
        <sec><title>Surgery</title>
          <sec><title>CVS</title><p>{words}</p><boxed-text><p>Recommendation 1.</p></boxed-text></sec>
        </sec>
        <sec><title>Search strategy</title><p>{words}</p></sec>
        <sec><title>Conflicts of interest</title><p>None.</p></sec>""")
    passages = chunk_article(path, META)
    sections = [p.section for p in passages]
    assert sections[0] == "Abstract"
    assert "Surgery > CVS" in sections
    assert not any("Search strategy" in s or "Conflicts" in s for s in sections)
    assert any("Recommendation 1." in p.text for p in passages)
    assert [p.id for p in passages] == [f"DOC-{i + 1:03d}" for i in range(len(passages))]


def test_long_paragraph_is_split_and_sections_never_mix(tmp_path):
    long = ". ".join(["sentence with five words"] * 200) + "."
    path = write(tmp_path, f"<sec><title>A</title><p>{long}</p></sec>"
                           f"<sec><title>B</title><p>tail</p></sec>")
    passages = chunk_article(path, META)
    assert all(len(p.text.split()) <= MAX_WORDS + 5 for p in passages)
    assert {p.section for p in passages if "tail" in p.text} == {"B"}
