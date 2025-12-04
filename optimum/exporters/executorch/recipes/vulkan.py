# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Dict, Union

from tabulate import tabulate
from torch.export import ExportedProgram

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    ExecutorchProgram,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from optimum.executorch.passes.remove_padding_idx_embedding_pass import RemovePaddingIdxEmbeddingPass

from ..integrations import (
    CausalLMExportableModule,
    MaskedLMExportableModule,
    MultiModalTextToTextExportableModule,
    Seq2SeqLMExportableModule,
)
from ..recipe_registry import register_recipe


@register_recipe("vulkan")
def export_to_executorch_with_vulkan(
    model: Union[
        CausalLMExportableModule,
        MaskedLMExportableModule,
        Seq2SeqLMExportableModule,
        MultiModalTextToTextExportableModule,
    ],
    **kwargs,
):
    """
    Export a PyTorch model to ExecuTorch w/ delegation to Vulkan backend.

    This function also writes metadata required by the ExecuTorch runtime to the model.
    Vulkan backend enables GPU acceleration on devices with Vulkan support (Android, Linux, etc.).

    Args:
        model (Union[CausalLMExportableModule, MaskedLMExportableModule, Seq2SeqLMExportableModule, MultiModalTextToTextExportableModule]):
            The PyTorch model to be exported to ExecuTorch.
        **kwargs:
            Additional keyword arguments for recipe-specific configurations:
                - require_dynamic_shapes (bool): Whether to require dynamic shape support (default: False)
                - skip_bool_tensors (bool): Whether to skip ops with bool tensors (default: False)

    Returns:
        Dict[str, ExecutorchProgram]:
            A map of exported and optimized program for ExecuTorch.
            For encoder-decoder models or multimodal models, it may generate multiple programs.
    """

    # Get Vulkan-specific options
    require_dynamic_shapes = kwargs.get("require_dynamic_shapes", False)
    skip_bool_tensors = kwargs.get("skip_bool_tensors", False)

    def _lower_to_executorch(
        exported_programs: Dict[str, ExportedProgram],
        metadata=None,
    ) -> Dict[str, ExecutorchProgram]:
        backend_config_dict = {
            "extract_delegate_segments": True,
            "memory_planning_pass": MemoryPlanningPass(alloc_graph_input=False),
            "do_quant_fusion_and_const_prop": True,
        }
        logging.debug(f"\nExported program: {exported_programs}")

        # If just one exported program, the method name in the .pte for it should be "forward".
        if len(exported_programs) == 1:
            exported_programs = {"forward": next(iter(exported_programs.values()))}

        # Configure Vulkan partitioner options
        vulkan_options = {
            "require_dynamic_shapes": require_dynamic_shapes,
            "skip_bool_tensors": skip_bool_tensors,
        }

        et_prog = to_edge_transform_and_lower(
            exported_programs,
            partitioner=[VulkanPartitioner(compile_options=vulkan_options)],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
            constant_methods=metadata,
            transform_passes=[RemovePaddingIdxEmbeddingPass()],
        )
        et_prog = et_prog.to_executorch(
            config=ExecutorchBackendConfig(**backend_config_dict),
        )
        pte_name = "model"
        for method in et_prog.methods:
            logging.debug(f"---------------------- Method: {method} ----------------------")
            logging.debug(f"\nExecuTorch program for {pte_name}.pte: {et_prog.exported_program(method).graph_module}")
            delegation_info = get_delegation_info(et_prog.exported_program(method).graph_module)
            logging.info(f"\nDelegation info Summary for {pte_name}.pte: {delegation_info.get_summary()}")
            logging.debug(
                f"\nDelegation info for {pte_name}.pte: {tabulate(delegation_info.get_operator_delegation_dataframe(), headers='keys', tablefmt='fancy_grid')}"
            )
        return {pte_name: et_prog}

    exported_progs = model.export()

    return _lower_to_executorch(exported_progs, model.metadata)

