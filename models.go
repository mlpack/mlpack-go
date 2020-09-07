package mlpack

/*
#include <capi/test_go_binding.h>
#include <capi/adaboost.h>
#include <capi/approx_kfn.h>
#include <capi/bayesian_linear_regression.h>
#include <capi/cf.h>
#include <capi/decision_stump.h>
#include <capi/decision_tree.h>
#include <capi/det.h>
#include <capi/fastmks.h>
#include <capi/gmm_train.h>
#include <capi/gmm_generate.h>
#include <capi/gmm_probability.h>
#include <capi/hmm_train.h>
#include <capi/hmm_loglik.h>
#include <capi/hmm_viterbi.h>
#include <capi/hmm_generate.h>
#include <capi/hoeffding_tree.h>
#include <capi/kde.h>
#include <capi/lars.h>
#include <capi/linear_regression.h>
#include <capi/linear_svm.h>
#include <capi/local_coordinate_coding.h>
#include <capi/logistic_regression.h>
#include <capi/lsh.h>
#include <capi/nbc.h>
#include <capi/knn.h>
#include <capi/kfn.h>
#include <capi/perceptron.h>
#include <capi/preprocess_scale.h>
#include <capi/random_forest.h>
#include <capi/krann.h>
#include <capi/softmax_regression.h>
#include <capi/sparse_coding.h>
*/
import "C"

import (
  "runtime"
  "unsafe"
)

type gaussianKernel struct {
  mem unsafe.Pointer 
}

func (m *gaussianKernel) allocGaussianKernel(identifier string) {
  m.mem = C.mlpackGetGaussianKernelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *gaussianKernel) getGaussianKernel(identifier string) {
  m.allocGaussianKernel(identifier)
}

func setGaussianKernel(identifier string, ptr *gaussianKernel) {
 C.mlpackSetGaussianKernelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type adaBoostModel struct {
  mem unsafe.Pointer 
}

func (m *adaBoostModel) allocAdaBoostModel(identifier string) {
  m.mem = C.mlpackGetAdaBoostModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *adaBoostModel) getAdaBoostModel(identifier string) {
  m.allocAdaBoostModel(identifier)
}

func setAdaBoostModel(identifier string, ptr *adaBoostModel) {
 C.mlpackSetAdaBoostModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type approxkfnModel struct {
  mem unsafe.Pointer 
}

func (m *approxkfnModel) allocApproxKFNModel(identifier string) {
  m.mem = C.mlpackGetApproxKFNModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *approxkfnModel) getApproxKFNModel(identifier string) {
  m.allocApproxKFNModel(identifier)
}

func setApproxKFNModel(identifier string, ptr *approxkfnModel) {
 C.mlpackSetApproxKFNModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type bayesianLinearRegression struct {
  mem unsafe.Pointer 
}

func (m *bayesianLinearRegression) allocBayesianLinearRegression(identifier string) {
  m.mem = C.mlpackGetBayesianLinearRegressionPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *bayesianLinearRegression) getBayesianLinearRegression(identifier string) {
  m.allocBayesianLinearRegression(identifier)
}

func setBayesianLinearRegression(identifier string, ptr *bayesianLinearRegression) {
 C.mlpackSetBayesianLinearRegressionPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type cfModel struct {
  mem unsafe.Pointer 
}

func (m *cfModel) allocCFModel(identifier string) {
  m.mem = C.mlpackGetCFModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *cfModel) getCFModel(identifier string) {
  m.allocCFModel(identifier)
}

func setCFModel(identifier string, ptr *cfModel) {
 C.mlpackSetCFModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type dsModel struct {
  mem unsafe.Pointer 
}

func (m *dsModel) allocDSModel(identifier string) {
  m.mem = C.mlpackGetDSModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *dsModel) getDSModel(identifier string) {
  m.allocDSModel(identifier)
}

func setDSModel(identifier string, ptr *dsModel) {
 C.mlpackSetDSModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type decisionTreeModel struct {
  mem unsafe.Pointer 
}

func (m *decisionTreeModel) allocDecisionTreeModel(identifier string) {
  m.mem = C.mlpackGetDecisionTreeModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *decisionTreeModel) getDecisionTreeModel(identifier string) {
  m.allocDecisionTreeModel(identifier)
}

func setDecisionTreeModel(identifier string, ptr *decisionTreeModel) {
 C.mlpackSetDecisionTreeModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type dTree struct {
  mem unsafe.Pointer 
}

func (m *dTree) allocDTree(identifier string) {
  m.mem = C.mlpackGetDTreePtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *dTree) getDTree(identifier string) {
  m.allocDTree(identifier)
}

func setDTree(identifier string, ptr *dTree) {
 C.mlpackSetDTreePtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type fastmksModel struct {
  mem unsafe.Pointer 
}

func (m *fastmksModel) allocFastMKSModel(identifier string) {
  m.mem = C.mlpackGetFastMKSModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *fastmksModel) getFastMKSModel(identifier string) {
  m.allocFastMKSModel(identifier)
}

func setFastMKSModel(identifier string, ptr *fastmksModel) {
 C.mlpackSetFastMKSModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type gmm struct {
  mem unsafe.Pointer 
}

func (m *gmm) allocGMM(identifier string) {
  m.mem = C.mlpackGetGMMPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *gmm) getGMM(identifier string) {
  m.allocGMM(identifier)
}

func setGMM(identifier string, ptr *gmm) {
 C.mlpackSetGMMPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type hmmModel struct {
  mem unsafe.Pointer 
}

func (m *hmmModel) allocHMMModel(identifier string) {
  m.mem = C.mlpackGetHMMModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *hmmModel) getHMMModel(identifier string) {
  m.allocHMMModel(identifier)
}

func setHMMModel(identifier string, ptr *hmmModel) {
 C.mlpackSetHMMModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type hoeffdingTreeModel struct {
  mem unsafe.Pointer 
}

func (m *hoeffdingTreeModel) allocHoeffdingTreeModel(identifier string) {
  m.mem = C.mlpackGetHoeffdingTreeModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *hoeffdingTreeModel) getHoeffdingTreeModel(identifier string) {
  m.allocHoeffdingTreeModel(identifier)
}

func setHoeffdingTreeModel(identifier string, ptr *hoeffdingTreeModel) {
 C.mlpackSetHoeffdingTreeModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type kdeModel struct {
  mem unsafe.Pointer 
}

func (m *kdeModel) allocKDEModel(identifier string) {
  m.mem = C.mlpackGetKDEModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *kdeModel) getKDEModel(identifier string) {
  m.allocKDEModel(identifier)
}

func setKDEModel(identifier string, ptr *kdeModel) {
 C.mlpackSetKDEModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type lars struct {
  mem unsafe.Pointer 
}

func (m *lars) allocLARS(identifier string) {
  m.mem = C.mlpackGetLARSPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *lars) getLARS(identifier string) {
  m.allocLARS(identifier)
}

func setLARS(identifier string, ptr *lars) {
 C.mlpackSetLARSPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type linearRegression struct {
  mem unsafe.Pointer 
}

func (m *linearRegression) allocLinearRegression(identifier string) {
  m.mem = C.mlpackGetLinearRegressionPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *linearRegression) getLinearRegression(identifier string) {
  m.allocLinearRegression(identifier)
}

func setLinearRegression(identifier string, ptr *linearRegression) {
 C.mlpackSetLinearRegressionPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type linearsvmModel struct {
  mem unsafe.Pointer 
}

func (m *linearsvmModel) allocLinearSVMModel(identifier string) {
  m.mem = C.mlpackGetLinearSVMModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *linearsvmModel) getLinearSVMModel(identifier string) {
  m.allocLinearSVMModel(identifier)
}

func setLinearSVMModel(identifier string, ptr *linearsvmModel) {
 C.mlpackSetLinearSVMModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type localCoordinateCoding struct {
  mem unsafe.Pointer 
}

func (m *localCoordinateCoding) allocLocalCoordinateCoding(identifier string) {
  m.mem = C.mlpackGetLocalCoordinateCodingPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *localCoordinateCoding) getLocalCoordinateCoding(identifier string) {
  m.allocLocalCoordinateCoding(identifier)
}

func setLocalCoordinateCoding(identifier string, ptr *localCoordinateCoding) {
 C.mlpackSetLocalCoordinateCodingPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type logisticRegression struct {
  mem unsafe.Pointer 
}

func (m *logisticRegression) allocLogisticRegression(identifier string) {
  m.mem = C.mlpackGetLogisticRegressionPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *logisticRegression) getLogisticRegression(identifier string) {
  m.allocLogisticRegression(identifier)
}

func setLogisticRegression(identifier string, ptr *logisticRegression) {
 C.mlpackSetLogisticRegressionPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type lshSearch struct {
  mem unsafe.Pointer 
}

func (m *lshSearch) allocLSHSearch(identifier string) {
  m.mem = C.mlpackGetLSHSearchPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *lshSearch) getLSHSearch(identifier string) {
  m.allocLSHSearch(identifier)
}

func setLSHSearch(identifier string, ptr *lshSearch) {
 C.mlpackSetLSHSearchPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type nbcModel struct {
  mem unsafe.Pointer 
}

func (m *nbcModel) allocNBCModel(identifier string) {
  m.mem = C.mlpackGetNBCModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *nbcModel) getNBCModel(identifier string) {
  m.allocNBCModel(identifier)
}

func setNBCModel(identifier string, ptr *nbcModel) {
 C.mlpackSetNBCModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type knnModel struct {
  mem unsafe.Pointer 
}

func (m *knnModel) allocKNNModel(identifier string) {
  m.mem = C.mlpackGetKNNModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *knnModel) getKNNModel(identifier string) {
  m.allocKNNModel(identifier)
}

func setKNNModel(identifier string, ptr *knnModel) {
 C.mlpackSetKNNModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type kfnModel struct {
  mem unsafe.Pointer 
}

func (m *kfnModel) allocKFNModel(identifier string) {
  m.mem = C.mlpackGetKFNModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *kfnModel) getKFNModel(identifier string) {
  m.allocKFNModel(identifier)
}

func setKFNModel(identifier string, ptr *kfnModel) {
 C.mlpackSetKFNModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type perceptronModel struct {
  mem unsafe.Pointer 
}

func (m *perceptronModel) allocPerceptronModel(identifier string) {
  m.mem = C.mlpackGetPerceptronModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *perceptronModel) getPerceptronModel(identifier string) {
  m.allocPerceptronModel(identifier)
}

func setPerceptronModel(identifier string, ptr *perceptronModel) {
 C.mlpackSetPerceptronModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type scalingModel struct {
  mem unsafe.Pointer 
}

func (m *scalingModel) allocScalingModel(identifier string) {
  m.mem = C.mlpackGetScalingModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *scalingModel) getScalingModel(identifier string) {
  m.allocScalingModel(identifier)
}

func setScalingModel(identifier string, ptr *scalingModel) {
 C.mlpackSetScalingModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type randomForestModel struct {
  mem unsafe.Pointer 
}

func (m *randomForestModel) allocRandomForestModel(identifier string) {
  m.mem = C.mlpackGetRandomForestModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *randomForestModel) getRandomForestModel(identifier string) {
  m.allocRandomForestModel(identifier)
}

func setRandomForestModel(identifier string, ptr *randomForestModel) {
 C.mlpackSetRandomForestModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type rannModel struct {
  mem unsafe.Pointer 
}

func (m *rannModel) allocRANNModel(identifier string) {
  m.mem = C.mlpackGetRANNModelPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *rannModel) getRANNModel(identifier string) {
  m.allocRANNModel(identifier)
}

func setRANNModel(identifier string, ptr *rannModel) {
 C.mlpackSetRANNModelPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type softmaxRegression struct {
  mem unsafe.Pointer 
}

func (m *softmaxRegression) allocSoftmaxRegression(identifier string) {
  m.mem = C.mlpackGetSoftmaxRegressionPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *softmaxRegression) getSoftmaxRegression(identifier string) {
  m.allocSoftmaxRegression(identifier)
}

func setSoftmaxRegression(identifier string, ptr *softmaxRegression) {
 C.mlpackSetSoftmaxRegressionPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

type sparseCoding struct {
  mem unsafe.Pointer 
}

func (m *sparseCoding) allocSparseCoding(identifier string) {
  m.mem = C.mlpackGetSparseCodingPtr(C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *sparseCoding) getSparseCoding(identifier string) {
  m.allocSparseCoding(identifier)
}

func setSparseCoding(identifier string, ptr *sparseCoding) {
 C.mlpackSetSparseCodingPtr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))
}

