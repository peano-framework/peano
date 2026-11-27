// This file is generated using cgeist Euler2D.cpp -S -I /cosma/apps/do009/dc-stok1/polygeist/lib/clang/18/include > euler2d.mlir
module {
  func.func @maxEigenvalueStateless(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: i32, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 5.000000e-01 : f64
    %cst_0 = arith.constant 0.39999999999999991 : f64
    %cst_1 = arith.constant 1.400000e+00 : f64
    %cst_2 = arith.constant 1.000000e+00 : f64
    %0 = affine.load %arg0[0] : memref<?xf64>
    %1 = arith.divf %cst_2, %0 : f64
    %2 = affine.load %arg0[3] : memref<?xf64>
    %3 = arith.mulf %1, %cst : f64
    %4 = affine.load %arg0[1] : memref<?xf64>
    %5 = arith.mulf %4, %4 : f64
    %6 = affine.load %arg0[2] : memref<?xf64>
    %7 = arith.mulf %6, %6 : f64
    %8 = arith.addf %5, %7 : f64
    %9 = arith.mulf %3, %8 : f64
    %10 = arith.subf %2, %9 : f64
    %11 = arith.mulf %10, %cst_0 : f64
    %12 = arith.mulf %11, %cst_1 : f64
    %13 = arith.mulf %12, %1 : f64
    %14 = arith.cmpi eq, %arg2, %c0_i32 : i32
    scf.if %14 {
      %15 = affine.load %arg0[1] : memref<?xf64>
      %16 = arith.mulf %15, %1 : f64
      %17 = arith.addf %16, %13 : f64
      %18 = math.absf %17 : f64
      %19 = arith.subf %16, %13 : f64
      %20 = math.absf %19 : f64
      %21 = func.call @fmax(%18, %20) : (f64, f64) -> f64
      affine.store %21, %arg3[0] : memref<?xf64>
    } else {
      %15 = arith.cmpi eq, %arg2, %c1_i32 : i32
      scf.if %15 {
        %16 = affine.load %arg0[2] : memref<?xf64>
        %17 = arith.mulf %16, %1 : f64
        %18 = arith.addf %17, %13 : f64
        %19 = math.absf %18 : f64
        %20 = arith.subf %17, %13 : f64
        %21 = math.absf %20 : f64
        %22 = func.call @fmax(%19, %21) : (f64, f64) -> f64
        affine.store %22, %arg3[0] : memref<?xf64>
      }
    }
    return
  }
  func.func private @fmax(f64, f64) -> f64 attributes {llvm.linkage = #llvm.linkage<external>}
  func.func @fluxStateless(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: i32, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 5.000000e-01 : f64
    %cst_0 = arith.constant 0.39999999999999991 : f64
    %cst_1 = arith.constant 1.000000e+00 : f64
    %0 = affine.load %arg0[0] : memref<?xf64>
    %1 = arith.divf %cst_1, %0 : f64
    %2 = affine.load %arg0[3] : memref<?xf64>
    %3 = arith.mulf %1, %cst : f64
    %4 = affine.load %arg0[1] : memref<?xf64>
    %5 = arith.mulf %4, %4 : f64
    %6 = affine.load %arg0[2] : memref<?xf64>
    %7 = arith.mulf %6, %6 : f64
    %8 = arith.addf %5, %7 : f64
    %9 = arith.mulf %3, %8 : f64
    %10 = arith.subf %2, %9 : f64
    %11 = arith.mulf %10, %cst_0 : f64
    %12 = arith.cmpi eq, %arg2, %c0_i32 : i32
    scf.if %12 {
      %13 = affine.load %arg0[1] : memref<?xf64>
      affine.store %13, %arg3[0] : memref<?xf64>
      %14 = affine.load %arg0[1] : memref<?xf64>
      %15 = arith.mulf %1, %14 : f64
      %16 = arith.mulf %15, %14 : f64
      %17 = arith.addf %16, %11 : f64
      affine.store %17, %arg3[1] : memref<?xf64>
      %18 = affine.load %arg0[1] : memref<?xf64>
      %19 = arith.mulf %1, %18 : f64
      %20 = affine.load %arg0[2] : memref<?xf64>
      %21 = arith.mulf %19, %20 : f64
      affine.store %21, %arg3[2] : memref<?xf64>
      %22 = affine.load %arg0[1] : memref<?xf64>
      %23 = arith.mulf %1, %22 : f64
      %24 = affine.load %arg0[3] : memref<?xf64>
      %25 = arith.addf %24, %11 : f64
      %26 = arith.mulf %23, %25 : f64
      affine.store %26, %arg3[3] : memref<?xf64>
    } else {
      %13 = arith.cmpi eq, %arg2, %c1_i32 : i32
      scf.if %13 {
        %14 = affine.load %arg0[2] : memref<?xf64>
        affine.store %14, %arg3[0] : memref<?xf64>
        %15 = affine.load %arg0[2] : memref<?xf64>
        %16 = arith.mulf %1, %15 : f64
        %17 = affine.load %arg0[1] : memref<?xf64>
        %18 = arith.mulf %16, %17 : f64
        affine.store %18, %arg3[1] : memref<?xf64>
        %19 = affine.load %arg0[2] : memref<?xf64>
        %20 = arith.mulf %1, %19 : f64
        %21 = arith.mulf %20, %19 : f64
        %22 = arith.addf %21, %11 : f64
        affine.store %22, %arg3[2] : memref<?xf64>
        %23 = affine.load %arg0[2] : memref<?xf64>
        %24 = arith.mulf %1, %23 : f64
        %25 = affine.load %arg0[3] : memref<?xf64>
        %26 = arith.addf %25, %11 : f64
        %27 = arith.mulf %24, %26 : f64
        affine.store %27, %arg3[3] : memref<?xf64>
      }
    }
    return
  }
}
