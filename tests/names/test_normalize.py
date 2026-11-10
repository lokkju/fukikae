from fukikae.names.normalize import NameClass, display_name, parse_name, suggest_aliases


def test_inner_voice_suffix_is_stripped_and_flagged():
    p = parse_name("WENDY/I")
    assert (p.key, p.inner, p.cls) == ("wendy", True, NameClass.NAMED)


def test_possession_spellings_share_one_key():
    assert (
        parse_name("MAKAROV(WRAITH)").key
        == parse_name("MAKAROV (WRAITH)").key
        == "makarov (wraith)"
    )


def test_slash_between_two_names_is_multi():
    assert parse_name("NATSU/ZEREF").cls is NameClass.MULTI
    assert parse_name("MAKAROV/WRAITH").cls is NameClass.MULTI


def test_role_labels():
    for raw in ("TOWNSPEOPLE A", "WIZARD B", "Guard 2"):
        assert parse_name(raw).cls is NameClass.ROLE


def test_role_label_with_a_trailing_period():
    assert parse_name("PRINCE A.").cls is NameClass.ROLE
    assert parse_name("GUARD 2.").cls is NameClass.ROLE
    assert parse_name("Mr.").cls is NameClass.NAMED


def test_sign_and_song_names_are_ignored_but_ed_is_a_speaker():
    assert parse_name("SIGN").cls is NameClass.IGNORED
    assert parse_name("Ed").cls is NameClass.NAMED


def test_width_and_case_fold_together():
    assert parse_name("ＮＡＴＳＵ").key == parse_name("Natsu").key == "natsu"


def test_display_title_cases_all_caps_only():
    assert display_name("YOUNG MAKAROV") == "Young Makarov"
    assert display_name("McGonagall") == "McGonagall"


def test_group_words_are_multi():
    assert parse_name("EVERYONE").cls is NameClass.MULTI


def test_typo_suggestion():
    assert suggest_aliases({"nebaru": 51, "nabaru": 3}) == [("nabaru", "nebaru")]
    assert suggest_aliases({"lucy": 80, "luce": 2}) == []  # under 5 characters


def test_eval_still_imports_moved_names():
    from fukikae.eval.groundtruth.base import canonical_name as c
    from fukikae.eval.groundtruth.base import check_alignment  # noqa: F401

    assert c("ERZA/I") == "ERZA"
