package mlpack

/*
#include <capi/approx_kfn.h>
#include <capi/bayesian_linear_regression.h>
#include <capi/cf.h>
#include <capi/decision_tree.h>
#include <capi/det.h>
#include <capi/fastmks.h>
#include <capi/gmm_train.h>
#include <capi/gmm_generate.h>
#include <capi/gmm_probability.h>
#include <capi/hmm_train.h>
#include <capi/hmm_generate.h>
#include <capi/hmm_loglik.h>
#include <capi/hmm_viterbi.h>
#include <capi/hoeffding_tree.h>
#include <capi/kde.h>
#include <capi/lars.h>
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
#include <capi/adaboost.h>
#include <capi/linear_regression.h>
*/
import "C"

import (
  "runtime"
  "unsafe"
)

type approxkfnModel struct {
  mem unsafe.Pointer 
}

func (m *approxkfnModel) allocApproxKFNModel(params *params, identifier string) {
  m.mem = C.mlpackGetApproxKFNModelPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *approxkfnModel) getApproxKFNModel(params *params, identifier string) {
  m.allocApproxKFNModel(params, identifier)
}

func setApproxKFNModel(params* params,
                           identifier string,
                           ptr *approxkfnModel) {
  C.mlpackSetApproxKFNModelPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type bayesianLinearRegression struct {
  mem unsafe.Pointer 
}

func (m *bayesianLinearRegression) allocBayesianLinearRegression(params *params, identifier string) {
  m.mem = C.mlpackGetBayesianLinearRegressionPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *bayesianLinearRegression) getBayesianLinearRegression(params *params, identifier string) {
  m.allocBayesianLinearRegression(params, identifier)
}

func setBayesianLinearRegression(params* params,
                           identifier string,
                           ptr *bayesianLinearRegression) {
  C.mlpackSetBayesianLinearRegressionPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type cfModel struct {
  mem unsafe.Pointer 
}

func (m *cfModel) allocCFModel(params *params, identifier string) {
  m.mem = C.mlpackGetCFModelPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *cfModel) getCFModel(params *params, identifier string) {
  m.allocCFModel(params, identifier)
}

func setCFModel(params* params,
                           identifier string,
                           ptr *cfModel) {
  C.mlpackSetCFModelPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type decisionTreeModel struct {
  mem unsafe.Pointer 
}

func (m *decisionTreeModel) allocDecisionTreeModel(params *params, identifier string) {
  m.mem = C.mlpackGetDecisionTreeModelPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *decisionTreeModel) getDecisionTreeModel(params *params, identifier string) {
  m.allocDecisionTreeModel(params, identifier)
}

func setDecisionTreeModel(params* params,
                           identifier string,
                           ptr *decisionTreeModel) {
  C.mlpackSetDecisionTreeModelPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type dTree struct {
  mem unsafe.Pointer 
}

func (m *dTree) allocDTree(params *params, identifier string) {
  m.mem = C.mlpackGetDTreePtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *dTree) getDTree(params *params, identifier string) {
  m.allocDTree(params, identifier)
}

func setDTree(params* params,
                           identifier string,
                           ptr *dTree) {
  C.mlpackSetDTreePtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type fastmksModel struct {
  mem unsafe.Pointer 
}

func (m *fastmksModel) allocFastMKSModel(params *params, identifier string) {
  m.mem = C.mlpackGetFastMKSModelPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *fastmksModel) getFastMKSModel(params *params, identifier string) {
  m.allocFastMKSModel(params, identifier)
}

func setFastMKSModel(params* params,
                           identifier string,
                           ptr *fastmksModel) {
  C.mlpackSetFastMKSModelPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type gmm struct {
  mem unsafe.Pointer 
}

func (m *gmm) allocGMM(params *params, identifier string) {
  m.mem = C.mlpackGetGMMPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *gmm) getGMM(params *params, identifier string) {
  m.allocGMM(params, identifier)
}

func setGMM(params* params,
                           identifier string,
                           ptr *gmm) {
  C.mlpackSetGMMPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type hmmModel struct {
  mem unsafe.Pointer 
}

func (m *hmmModel) allocHMMModel(params *params, identifier string) {
  m.mem = C.mlpackGetHMMModelPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *hmmModel) getHMMModel(params *params, identifier string) {
  m.allocHMMModel(params, identifier)
}

func setHMMModel(params* params,
                           identifier string,
                           ptr *hmmModel) {
  C.mlpackSetHMMModelPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type hoeffdingTreeModel struct {
  mem unsafe.Pointer 
}

func (m *hoeffdingTreeModel) allocHoeffdingTreeModel(params *params, identifier string) {
  m.mem = C.mlpackGetHoeffdingTreeModelPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *hoeffdingTreeModel) getHoeffdingTreeModel(params *params, identifier string) {
  m.allocHoeffdingTreeModel(params, identifier)
}

func setHoeffdingTreeModel(params* params,
                           identifier string,
                           ptr *hoeffdingTreeModel) {
  C.mlpackSetHoeffdingTreeModelPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type kdeModel struct {
  mem unsafe.Pointer 
}

func (m *kdeModel) allocKDEModel(params *params, identifier string) {
  m.mem = C.mlpackGetKDEModelPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *kdeModel) getKDEModel(params *params, identifier string) {
  m.allocKDEModel(params, identifier)
}

func setKDEModel(params* params,
                           identifier string,
                           ptr *kdeModel) {
  C.mlpackSetKDEModelPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type lars struct {
  mem unsafe.Pointer 
}

func (m *lars) allocLARS(params *params, identifier string) {
  m.mem = C.mlpackGetLARSPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *lars) getLARS(params *params, identifier string) {
  m.allocLARS(params, identifier)
}

func setLARS(params* params,
                           identifier string,
                           ptr *lars) {
  C.mlpackSetLARSPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type linearsvmModel struct {
  mem unsafe.Pointer 
}

func (m *linearsvmModel) allocLinearSVMModel(params *params, identifier string) {
  m.mem = C.mlpackGetLinearSVMModelPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *linearsvmModel) getLinearSVMModel(params *params, identifier string) {
  m.allocLinearSVMModel(params, identifier)
}

func setLinearSVMModel(params* params,
                           identifier string,
                           ptr *linearsvmModel) {
  C.mlpackSetLinearSVMModelPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type localCoordinateCoding struct {
  mem unsafe.Pointer 
}

func (m *localCoordinateCoding) allocLocalCoordinateCoding(params *params, identifier string) {
  m.mem = C.mlpackGetLocalCoordinateCodingPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *localCoordinateCoding) getLocalCoordinateCoding(params *params, identifier string) {
  m.allocLocalCoordinateCoding(params, identifier)
}

func setLocalCoordinateCoding(params* params,
                           identifier string,
                           ptr *localCoordinateCoding) {
  C.mlpackSetLocalCoordinateCodingPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type logisticRegression struct {
  mem unsafe.Pointer 
}

func (m *logisticRegression) allocLogisticRegression(params *params, identifier string) {
  m.mem = C.mlpackGetLogisticRegressionPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *logisticRegression) getLogisticRegression(params *params, identifier string) {
  m.allocLogisticRegression(params, identifier)
}

func setLogisticRegression(params* params,
                           identifier string,
                           ptr *logisticRegression) {
  C.mlpackSetLogisticRegressionPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type lshSearch struct {
  mem unsafe.Pointer 
}

func (m *lshSearch) allocLSHSearch(params *params, identifier string) {
  m.mem = C.mlpackGetLSHSearchPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *lshSearch) getLSHSearch(params *params, identifier string) {
  m.allocLSHSearch(params, identifier)
}

func setLSHSearch(params* params,
                           identifier string,
                           ptr *lshSearch) {
  C.mlpackSetLSHSearchPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type nbcModel struct {
  mem unsafe.Pointer 
}

func (m *nbcModel) allocNBCModel(params *params, identifier string) {
  m.mem = C.mlpackGetNBCModelPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *nbcModel) getNBCModel(params *params, identifier string) {
  m.allocNBCModel(params, identifier)
}

func setNBCModel(params* params,
                           identifier string,
                           ptr *nbcModel) {
  C.mlpackSetNBCModelPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type knnModel struct {
  mem unsafe.Pointer 
}

func (m *knnModel) allocKNNModel(params *params, identifier string) {
  m.mem = C.mlpackGetKNNModelPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *knnModel) getKNNModel(params *params, identifier string) {
  m.allocKNNModel(params, identifier)
}

func setKNNModel(params* params,
                           identifier string,
                           ptr *knnModel) {
  C.mlpackSetKNNModelPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type kfnModel struct {
  mem unsafe.Pointer 
}

func (m *kfnModel) allocKFNModel(params *params, identifier string) {
  m.mem = C.mlpackGetKFNModelPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *kfnModel) getKFNModel(params *params, identifier string) {
  m.allocKFNModel(params, identifier)
}

func setKFNModel(params* params,
                           identifier string,
                           ptr *kfnModel) {
  C.mlpackSetKFNModelPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type perceptronModel struct {
  mem unsafe.Pointer 
}

func (m *perceptronModel) allocPerceptronModel(params *params, identifier string) {
  m.mem = C.mlpackGetPerceptronModelPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *perceptronModel) getPerceptronModel(params *params, identifier string) {
  m.allocPerceptronModel(params, identifier)
}

func setPerceptronModel(params* params,
                           identifier string,
                           ptr *perceptronModel) {
  C.mlpackSetPerceptronModelPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type scalingModel struct {
  mem unsafe.Pointer 
}

func (m *scalingModel) allocScalingModel(params *params, identifier string) {
  m.mem = C.mlpackGetScalingModelPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *scalingModel) getScalingModel(params *params, identifier string) {
  m.allocScalingModel(params, identifier)
}

func setScalingModel(params* params,
                           identifier string,
                           ptr *scalingModel) {
  C.mlpackSetScalingModelPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type randomForestModel struct {
  mem unsafe.Pointer 
}

func (m *randomForestModel) allocRandomForestModel(params *params, identifier string) {
  m.mem = C.mlpackGetRandomForestModelPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *randomForestModel) getRandomForestModel(params *params, identifier string) {
  m.allocRandomForestModel(params, identifier)
}

func setRandomForestModel(params* params,
                           identifier string,
                           ptr *randomForestModel) {
  C.mlpackSetRandomForestModelPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type raModel struct {
  mem unsafe.Pointer 
}

func (m *raModel) allocRAModel(params *params, identifier string) {
  m.mem = C.mlpackGetRAModelPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *raModel) getRAModel(params *params, identifier string) {
  m.allocRAModel(params, identifier)
}

func setRAModel(params* params,
                           identifier string,
                           ptr *raModel) {
  C.mlpackSetRAModelPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type softmaxRegression struct {
  mem unsafe.Pointer 
}

func (m *softmaxRegression) allocSoftmaxRegression(params *params, identifier string) {
  m.mem = C.mlpackGetSoftmaxRegressionPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *softmaxRegression) getSoftmaxRegression(params *params, identifier string) {
  m.allocSoftmaxRegression(params, identifier)
}

func setSoftmaxRegression(params* params,
                           identifier string,
                           ptr *softmaxRegression) {
  C.mlpackSetSoftmaxRegressionPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type sparseCoding struct {
  mem unsafe.Pointer 
}

func (m *sparseCoding) allocSparseCoding(params *params, identifier string) {
  m.mem = C.mlpackGetSparseCodingPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *sparseCoding) getSparseCoding(params *params, identifier string) {
  m.allocSparseCoding(params, identifier)
}

func setSparseCoding(params* params,
                           identifier string,
                           ptr *sparseCoding) {
  C.mlpackSetSparseCodingPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type adaBoostModel struct {
  mem unsafe.Pointer 
}

func (m *adaBoostModel) allocAdaBoostModel(params *params, identifier string) {
  m.mem = C.mlpackGetAdaBoostModelPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *adaBoostModel) getAdaBoostModel(params *params, identifier string) {
  m.allocAdaBoostModel(params, identifier)
}

func setAdaBoostModel(params* params,
                           identifier string,
                           ptr *adaBoostModel) {
  C.mlpackSetAdaBoostModelPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

type linearRegression struct {
  mem unsafe.Pointer 
}

func (m *linearRegression) allocLinearRegression(params *params, identifier string) {
  m.mem = C.mlpackGetLinearRegressionPtr(params.mem,
      C.CString(identifier))
  runtime.KeepAlive(m)
}

func (m *linearRegression) getLinearRegression(params *params, identifier string) {
  m.allocLinearRegression(params, identifier)
}

func setLinearRegression(params* params,
                           identifier string,
                           ptr *linearRegression) {
  C.mlpackSetLinearRegressionPtr(params.mem,
      C.CString(identifier), ptr.mem)
}

