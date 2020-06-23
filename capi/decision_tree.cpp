#include </home/ryan/src/mlpack-3.3.2/src/mlpack/methods/decision_tree/decision_tree_main.cpp>
#include <mlpack/bindings/go/mlpack/capi/cli_util.hpp>

using namespace mlpack;
using namespace mlpack::util;
using namespace std;

extern "C" void mlpackSetDecisionTreeModelPtr(
    const char* identifier, 
    void* value)
{
  SetParamPtr<DecisionTreeModel>(identifier,
      static_cast<DecisionTreeModel*>(value));
}

extern "C" void *mlpackGetDecisionTreeModelPtr(const char* identifier)
{
  DecisionTreeModel *modelptr = GetParamPtr<DecisionTreeModel>(identifier);
  return modelptr;
}

static void DecisionTreeMlpackMain()
{
  mlpackMain();
}

extern "C" void mlpackDecisionTree()
{
  DecisionTreeMlpackMain();
}

