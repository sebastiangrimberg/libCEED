// Copyright (c) 2017-2022, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed.h>
#include <ceed/backend.h>
#include <libxsmm.h>

#include "ceed-xsmm.h"

//------------------------------------------------------------------------------
// Tensor Contract Apply
//------------------------------------------------------------------------------
static int CeedTensorContractApply_Xsmm(CeedTensorContract contract, CeedInt A, CeedInt B, CeedInt C, CeedInt J, const double *restrict t,
                                        CeedTransposeMode t_mode, const CeedInt add, const double *restrict u, double *restrict v) {
  if (C == 1) {
    double alpha = 1.0, beta = 1.0;
    char   trans_u = 'N', trans_t = 'N';
    if (t_mode == CEED_NOTRANSPOSE) trans_t = 'T';
    if (!add) beta = 0.0;

    // LIBXSMM GEMM
    libxsmm_dgemm(&trans_t, &trans_u, &J, &A, &B, &alpha, &t[0], NULL, &u[0], NULL, &beta, &v[0], NULL);

    return CEED_ERROR_SUCCESS;
  } else {
    double alpha = 1.0, beta = 1.0;
    char   trans_u = 'N', trans_t = 'N';
    if (t_mode == CEED_TRANSPOSE) trans_t = 'T';
    if (!add) beta = 0.0;

    // LIBXSMM GEMM
    for (CeedInt a = 0; a < A; a++) {
      libxsmm_dgemm(&trans_u, &trans_t, &C, &J, &B, &alpha, &u[a * B * C], NULL, &t[0], NULL, &beta, &v[a * J * C], NULL);
    }

    return CEED_ERROR_SUCCESS;
  }
}

//------------------------------------------------------------------------------
// Tensor Contract Create
//------------------------------------------------------------------------------
int CeedTensorContractCreate_f64_Xsmm(CeedBasis basis, CeedTensorContract contract) {
  Ceed ceed;
  CeedCallBackend(CeedTensorContractGetCeed(contract, &ceed));

  CeedCallBackend(CeedSetBackendFunction(ceed, "TensorContract", contract, "Apply", CeedTensorContractApply_Xsmm));

  return CEED_ERROR_SUCCESS;
}

//------------------------------------------------------------------------------
