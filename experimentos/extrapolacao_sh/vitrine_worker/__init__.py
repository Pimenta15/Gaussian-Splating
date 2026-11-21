"""Subconjunto do pipeline de reconstrução usado pelos experimentos de extrapolação em elevação.

Cópia fiel dos módulos datasets, export, metrics, sh_observado, train_3dgs e train_refgs do pacote
vitrine_worker (commit 634111e), incluída aqui para que os scripts de experimentos/extrapolacao_sh rodem
sem o repositório do serviço. train_carro (só usado por carro_real.py, que depende de dados próprios)
não está incluído.
"""
