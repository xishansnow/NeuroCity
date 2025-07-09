"""
Instant NGP Python 3.10 兼容性测试

专门测试 Instant NGP 实现与 Python 3.10 的兼容性，
重点检查内置容器类型的使用和现代 Python 特性。
"""

from __future__ import annotations

import pytest
import sys
import torch
import numpy as np

from dataclasses import asdict, is_dataclass
import os

# 添加 src 目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from nerfs.instant_ngp.core import (
    InstantNGPConfig,
    InstantNGPModel,
    HashEncoder,
    SHEncoder,
    InstantNGPLoss,
)
from nerfs.instant_ngp.trainer import InstantNGPTrainer, InstantNGPTrainerConfig
from nerfs.instant_ngp.renderer import (
    InstantNGPRenderer as InstantNGPInferenceRenderer,
    InstantNGPRendererConfig,
)
from nerfs.instant_ngp.dataset import (
    InstantNGPDatasetConfig,
)


class TestPython310Compatibility:
    """Python 3.10 兼容性测试类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_python_version(self):
        """测试 Python 版本"""
        version_info = sys.version_info
        print(f"Python 版本: {version_info.major}.{version_info.minor}.{version_info.micro}")

        # 确保是 Python 3.10 或更高版本
        assert version_info >= (3, 10), f"需要 Python 3.10+，当前版本: {version_info}"

    def test_builtin_container_annotations(self):
        """测试内置容器类型注解的使用"""
        # 这些应该在 Python 3.10+ 中正常工作，不需要从 typing 导入

        # 测试 dict 类型注解
        config_dict: dict[str, Any] = {"key": "value", "number": 42}
        assert isinstance(config_dict, dict)

        # 测试 list 类型注解
        int_list: list[int] = [1, 2, 3, 4, 5]
        assert isinstance(int_list, list)
        assert all(isinstance(x, int) for x in int_list)

        # 测试 tuple 类型注解
        coord_tuple: tuple[float, float, float] = (1.0, 2.0, 3.0)
        assert isinstance(coord_tuple, tuple)
        assert len(coord_tuple) == 3

        # 测试嵌套容器类型
        nested_dict: dict[str, list[float]] = {
            "positions": [1.0, 2.0, 3.0],
            "colors": [0.5, 0.7, 0.9],
        }
        assert isinstance(nested_dict, dict)
        assert all(isinstance(v, list) for v in nested_dict.values())

    def test_dataclass_compatibility(self):
        """测试数据类与 Python 3.10 的兼容性"""
        # 测试配置数据类
        config = InstantNGPConfig()

        # 检查是否是数据类
        assert is_dataclass(config)

        # 测试转换为字典（使用内置类型）
        config_dict: dict[str, Any] = asdict(config)
        assert isinstance(config_dict, dict)

        # 检查关键字段
        expected_fields: list[str] = [
            "num_levels",
            "base_resolution",
            "finest_resolution",
            "hidden_dim",
            "use_amp",
        ]

        for field in expected_fields:
            assert field in config_dict

        # 测试其他配置类
        trainer_config = InstantNGPTrainerConfig()
        trainer_dict: dict[str, Any] = asdict(trainer_config)
        assert isinstance(trainer_dict, dict)

        renderer_config = InstantNGPRendererConfig()
        renderer_dict: dict[str, Any] = asdict(renderer_config)
        assert isinstance(renderer_dict, dict)

    def test_type_hints_compatibility(self):
        """测试类型提示的兼容性"""

        # 测试函数参数和返回值类型提示
        def process_tensor_dict(data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            """使用现代类型提示的函数"""
            result: dict[str, torch.Tensor] = {}
            for key, tensor in data.items():
                result[key] = tensor * 2
            return result

        # 测试函数
        test_data: dict[str, torch.Tensor] = {
            "positions": torch.randn(10, 3),
            "colors": torch.rand(10, 3),
        }

        result = process_tensor_dict(test_data)
        assert isinstance(result, dict)
        assert len(result) == len(test_data)

    def test_union_types_with_pipe_operator(self):
        """测试使用管道操作符的联合类型（Python 3.10+ 特性）"""
        # 这个特性在 Python 3.10+ 中可用
        if sys.version_info >= (3, 10):
            # 测试简单联合类型
            def process_value(value: int | float) -> float:
                return float(value)

            assert process_value(42) == 42.0
            assert process_value(3.14) == 3.14

            # 测试复杂联合类型
            def process_data(data: dict[str, int | float]) -> dict[str, float]:
                return {k: float(v) for k, v in data.items()}

            test_data = {"a": 1, "b": 2.5, "c": 3}
            result = process_data(test_data)
            assert all(isinstance(v, float) for v in result.values())

    def test_instant_ngp_components_with_modern_types(self):
        """测试 Instant NGP 组件使用现代类型注解"""
        # 测试模型配置
        config = InstantNGPConfig()

        # 验证配置中的类型
        assert isinstance(config.num_levels, int)
        assert isinstance(config.base_resolution, int)
        assert isinstance(config.use_amp, bool)

        # 测试模型初始化
        model = InstantNGPModel(config)

        # 测试前向传播（返回现代类型注解的字典）
        batch_size = 100
        positions = torch.randn(batch_size, 3, device=self.device)
        directions = torch.randn(batch_size, 3, device=self.device)
        directions = torch.nn.functional.normalize(directions, dim=-1)

        with torch.no_grad():
            output: dict[str, torch.Tensor] = model(positions, directions)

        # 验证输出类型
        assert isinstance(output, dict)
        assert "density" in output
        assert "color" in output
        assert isinstance(output["density"], torch.Tensor)
        assert isinstance(output["color"], torch.Tensor)

    def test_hash_encoder_with_modern_types(self):
        """测试哈希编码器使用现代类型"""
        config = InstantNGPConfig(num_levels=4)
        encoder = HashEncoder(config)

        # 测试分辨率列表（使用现代类型注解）
        resolutions: list[int] = encoder.resolutions
        assert isinstance(resolutions, list)
        assert all(isinstance(res, int) for res in resolutions)
        assert len(resolutions) == config.num_levels

        # 测试编码器输出
        positions = torch.randn(50, 3, device=self.device)
        encoded = encoder(positions)

        assert isinstance(encoded, torch.Tensor)
        assert encoded.shape[1] == encoder.output_dim

    def test_trainer_with_modern_types(self):
        """测试训练器使用现代类型"""
        model_config = InstantNGPConfig(num_levels=2, hidden_dim=16)
        trainer_config = InstantNGPTrainerConfig(batch_size=64)

        model = InstantNGPModel(model_config)
        trainer = InstantNGPTrainer(model, trainer_config, device=self.device)

        # 测试损失历史（使用现代类型）
        loss_history: list[float] = trainer.loss_history
        assert isinstance(loss_history, list)

        # 执行训练步骤
        batch_data: dict[str, torch.Tensor] = {
            "ray_origins": torch.randn(50, 3, device=self.device),
            "ray_directions": torch.nn.functional.normalize(
                torch.randn(50, 3, device=self.device), dim=-1
            ),
            "colors": torch.rand(50, 3, device=self.device),
        }

        loss_info: dict[str, torch.Tensor] = trainer.training_step(batch_data)

        # 验证返回类型
        assert isinstance(loss_info, dict)
        assert "total_loss" in loss_info
        assert isinstance(loss_info["total_loss"], torch.Tensor)

    def test_renderer_with_modern_types(self):
        """测试渲染器使用现代类型"""
        model_config = InstantNGPConfig(num_levels=2, hidden_dim=16)
        renderer_config = InstantNGPRendererConfig(image_width=32, image_height=32)

        model = InstantNGPModel(model_config).to(self.device)
        renderer = InstantNGPInferenceRenderer(model, renderer_config)

        # 测试相机内参（使用现代字典类型）
        intrinsics: dict[str, float] = {
            "fx": 100.0,
            "fy": 100.0,
            "cx": 16.0,
            "cy": 16.0,
        }

        renderer.set_camera_intrinsics(intrinsics)

        # 测试渲染输出
        ray_origins = torch.randn(100, 3, device=self.device)
        ray_directions = torch.nn.functional.normalize(
            torch.randn(100, 3, device=self.device), dim=-1
        )

        with torch.no_grad():
            render_output: dict[str, torch.Tensor] = renderer.render_rays(
                ray_origins, ray_directions
            )

        # 验证输出类型
        assert isinstance(render_output, dict)
        expected_keys: list[str] = ["color", "depth"]
        for key in expected_keys:
            if key in render_output:
                assert isinstance(render_output[key], torch.Tensor)

    def test_container_operations_compatibility(self):
        """测试容器操作的兼容性"""
        # 测试字典操作
        config_data: dict[str, Any] = {
            "model": {"hidden_dim": 64, "num_layers": 3},
            "training": {"batch_size": 1024, "learning_rate": 1e-2},
            "rendering": {"image_size": (256, 256), "num_samples": 128},
        }

        # 字典推导式
        model_params: dict[str, int] = {
            k: v for k, v in config_data["model"].items() if isinstance(v, int)
        }
        assert isinstance(model_params, dict)

        # 列表操作
        batch_sizes: list[int] = [64, 128, 256, 512, 1024]
        large_batches: list[int] = [size for size in batch_sizes if size >= 256]
        assert isinstance(large_batches, list)
        assert len(large_batches) == 3

        # 集合操作
        available_features: set[str] = {"hash_encoding", "sh_encoding", "volume_rendering"}
        required_features: set[str] = {"hash_encoding", "volume_rendering"}
        missing_features: set[str] = required_features - available_features
        assert isinstance(missing_features, set)
        assert len(missing_features) == 0

    def test_pathlib_compatibility(self):
        """测试 pathlib 兼容性"""
        from pathlib import Path
        import tempfile

        # 创建临时路径
        temp_dir = Path(tempfile.mkdtemp())

        # 测试路径操作
        model_path: Path = temp_dir / "model.pth"
        config_path: Path = temp_dir / "config.json"

        assert isinstance(model_path, Path)
        assert isinstance(config_path, Path)

        # 清理
        import shutil

        shutil.rmtree(temp_dir)

    def test_modern_exception_handling(self):
        """测试现代异常处理"""
        # 测试异常组（Python 3.11+ 特性，但确保向后兼容）
        try:
            # 测试无效配置
            invalid_config = InstantNGPConfig(num_levels=0)
            invalid_config.__post_init__()
        except AssertionError as e:
            assert isinstance(e, AssertionError)
        except Exception as e:
            # 确保其他异常也能正确处理
            assert isinstance(e, Exception)

    def test_dataclass_field_access(self):
        """测试数据类字段访问"""
        from dataclasses import fields

        # 测试配置字段
        config = InstantNGPConfig()
        config_fields = fields(config)

        # 获取字段名列表（使用现代类型）
        field_names: list[str] = [f.name for f in config_fields]
        field_types: dict[str, type] = {f.name: f.type for f in config_fields}

        assert isinstance(field_names, list)
        assert isinstance(field_types, dict)
        assert "num_levels" in field_names
        assert "hidden_dim" in field_names

    def test_tensor_operations_with_modern_types(self):
        """测试张量操作使用现代类型"""
        # 创建张量字典
        tensors: dict[str, torch.Tensor] = {
            "positions": torch.randn(100, 3, device=self.device),
            "directions": torch.randn(100, 3, device=self.device),
            "colors": torch.rand(100, 3, device=self.device),
        }

        # 张量操作
        normalized_tensors: dict[str, torch.Tensor] = {}
        for key, tensor in tensors.items():
            if key == "directions":
                normalized_tensors[key] = torch.nn.functional.normalize(tensor, dim=-1)
            else:
                normalized_tensors[key] = tensor

        # 验证类型
        assert isinstance(normalized_tensors, dict)
        for key, tensor in normalized_tensors.items():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.device == self.device

    def teardown_method(self):
        """每个测试方法后的清理"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TestPython310SpecificFeatures:
    """Python 3.10 特定特性测试"""

    def test_structural_pattern_matching(self):
        """测试结构化模式匹配（Python 3.10+ 特性）"""
        if sys.version_info >= (3, 10):

            def process_config_value(value: Any) -> str:
                match value:
                    case int() if value > 0:
                        return f"positive_int_{value}"
                    case float() if value > 0.0:
                        return f"positive_float_{value}"
                    case str() if len(value) > 0:
                        return f"string_{value}"
                    case bool():
                        return f"boolean_{value}"
                    case _:
                        return "unknown"

            # 测试不同类型的值
            assert process_config_value(42) == "positive_int_42"
            assert process_config_value(3.14) == "positive_float_3.14"
            assert process_config_value("test") == "string_test"
            assert process_config_value(True) == "boolean_True"
        else:
            pytest.skip("Structural pattern matching requires Python 3.10+")

    def test_parenthesized_context_managers(self):
        """测试带括号的上下文管理器（Python 3.10+ 特性）"""
        import tempfile
        import os

        # 创建临时文件进行测试
        with (
            tempfile.NamedTemporaryFile(mode="w", delete=False) as f1,
            tempfile.NamedTemporaryFile(mode="w", delete=False) as f2,
        ):
            f1.write("test1")
            f2.write("test2")
            f1_name = f1.name
            f2_name = f2.name

        # 验证文件创建成功
        assert os.path.exists(f1_name)
        assert os.path.exists(f2_name)

        # 清理
        os.unlink(f1_name)
        os.unlink(f2_name)

    def test_precise_error_locations(self):
        """测试精确错误位置（Python 3.10+ 改进）"""
        # Python 3.10+ 提供了更精确的错误位置信息
        try:
            # 故意创建一个错误
            config = InstantNGPConfig()
            invalid_access = config.nonexistent_attribute
        except AttributeError as e:
            # 检查错误信息
            error_msg = str(e)
            assert "nonexistent_attribute" in error_msg
            assert isinstance(e, AttributeError)


if __name__ == "__main__":
    pytest.main([__file__])
