# -*- coding: utf-8 -*-
"""元数据筛选检索的纯函数测试（不碰 GPU / 索引 / qdrant）。"""
import rag_core.retriever as R


class TestNormalizeFilters:
    def test_none_and_empty(self):
        assert R.normalize_filters(None) is None
        assert R.normalize_filters({}) is None
        assert R.normalize_filters({"year_min": None, "authors": []}) is None

    def test_year_int_coercion(self):
        f = R.normalize_filters({"year_min": "2020", "year_max": 2024})
        assert f == {"year_min": 2020, "year_max": 2024}

    def test_list_cleaning(self):
        f = R.normalize_filters({"methods": ["随机森林", "", None, " xgboost "]})
        assert f == {"methods": ["随机森林", "xgboost"]}

    def test_single_string_to_list(self):
        assert R.normalize_filters({"tasks": "信贷风控"}) == {"tasks": ["信贷风控"]}


class TestMatchMetaFilter:
    META = {
        "year": 2022, "author": "李安哲",
        "methods": ["传统机器学习", "集成学习"], "tasks": ["信贷风控"],
    }

    def test_no_filters(self):
        assert R.match_meta_filter(self.META, None) is True
        assert R.match_meta_filter({}, {}) is True

    def test_year_bounds(self):
        assert R.match_meta_filter(self.META, {"year_min": 2022}) is True
        assert R.match_meta_filter(self.META, {"year_max": 2021}) is False
        # 块缺少年份时，年份过滤不通过（宁可漏，不可错放）
        assert R.match_meta_filter({"author": "x"}, {"year_min": 2020}) is False

    def test_author_substring(self):
        assert R.match_meta_filter(self.META, {"authors": ["李安"]}) is True
        assert R.match_meta_filter(self.META, {"authors": ["王五"]}) is False

    def test_methods_any_hit(self):
        assert R.match_meta_filter(self.META, {"methods": ["随机森林", "集成学习"]}) is True
        # 双向包含："集成" 命中 "集成学习"
        assert R.match_meta_filter(self.META, {"methods": ["集成"]}) is True
        assert R.match_meta_filter(self.META, {"methods": ["深度学习"]}) is False

    def test_tasks(self):
        assert R.match_meta_filter(self.META, {"tasks": ["信贷风控", "交易策略"]}) is True
        assert R.match_meta_filter(self.META, {"tasks": ["债券"]}) is False

    def test_combined(self):
        f = {"year_min": 2022, "methods": ["集成学习"], "tasks": ["信贷风控"]}
        assert R.match_meta_filter(self.META, f) is True
        f2 = {"year_min": 2023, "methods": ["集成学习"], "tasks": ["信贷风控"]}
        assert R.match_meta_filter(self.META, f2) is False

    def test_string_field_tolerance(self):
        meta = {"year": 2020, "author": "李安哲", "methods": "随机森林", "tasks": []}
        assert R.match_meta_filter(meta, {"methods": ["随机森林"]}) is True
