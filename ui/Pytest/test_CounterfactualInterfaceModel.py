# Author: Moises Henrique Pereira
# this class handle the model's functions tests  

import pytest

from ui.mainTest import StaticObjects

# the function openChosenDataset expect a str type
# send another type as parameter would arrise an assertionError
@pytest.mark.parametrize('chosenDataset', [1, 2.9, ['str'], False, ('t1', 't2'), None])
def test_CIM_openChosenDataset_wrong_parameter(chosenDataset):
    with pytest.raises(AssertionError):
        counterfactualInterfaceModel = StaticObjects.staticCounterfactualInterfaceModel()
        counterfactualInterfaceModel.openChosenDataset(chosenDataset)

# the function openChosenDataset expect a not empty str type
# send an empty str type as parameter would arrise an assertionError
def test_CIM_openChosenDataset_empty_str():
    with pytest.raises(AssertionError):
        counterfactualInterfaceModel = StaticObjects.staticCounterfactualInterfaceModel()
        counterfactualInterfaceModel.openChosenDataset('')

# the function openChosenDataset expect a not empty str type
# a valid str would not arrise assertionError
def test_CIM_openChosenDataset_right_parameter():
    counterfactualInterfaceModel = StaticObjects.staticCounterfactualInterfaceModel()
    counterfactualInterfaceModel.openChosenDataset('German-Credit')

# the function transformDataPoint expect a not empty array
# send none as data point would arrise an assertionError
def test_CIM_transformDataPoint_none_parameter():
    with pytest.raises(AssertionError):
        counterfactualInterfaceModel = StaticObjects.staticCounterfactualInterfaceModel()
        counterfactualInterfaceModel.transformDataPoint(None)

# the function transformDataPoint expect a not empty array, and valid length
# send an invalid length as data point would arrise an assertionError
def test_CIM_transformDataPoint_wrong_length():
    with pytest.raises(AssertionError):
        counterfactualInterfaceModel = StaticObjects.staticCounterfactualInterfaceModel()
        counterfactualInterfaceModel.openChosenDataset('German-Credit')
        counterfactualInterfaceModel.transformDataPoint([67,'male',2,'own',0,1,1169,6])

# the function transformDataPoint expect a not empty array, and it with valid length
# would not arrise assertionError
def test_CIM_transformDataPoint_right_parameter():
    counterfactualInterfaceModel = StaticObjects.staticCounterfactualInterfaceModel()
    counterfactualInterfaceModel.openChosenDataset('German-Credit')
    counterfactualInterfaceModel.transformDataPoint([67,'male',2,'own',0,1,1169,6,'radio/TV'])

# the function transformSingleNumericalValue expect a string as feature name, and a number
# send another type as feature would arrise assertionError
@pytest.mark.parametrize('feature', [1, 2.9, False, ('t1', 't2'), None])
def test_CIM_transformSingleNumericalValue_feature_wrong_type(feature):
    with pytest.raises(AssertionError):
        counterfactualInterfaceModel = StaticObjects.staticCounterfactualInterfaceModel()
        counterfactualInterfaceModel.openChosenDataset('German-Credit')
        counterfactualInterfaceModel.transformSingleNumericalValue(feature, 67)

# the function transformSingleNumericalValue expect a string as feature name, and a number
# send none as value would arrise assertionError
def test_CIM_transformSingleNumericalValue_value_none_parameter():
    with pytest.raises(AssertionError):
        counterfactualInterfaceModel = StaticObjects.staticCounterfactualInterfaceModel()
        counterfactualInterfaceModel.openChosenDataset('German-Credit')
        counterfactualInterfaceModel.transformSingleNumericalValue('Age', None)

# the function transformSingleNumericalValue expect a string as feature name, and a number
# send these would not arrise assertionError
def test_CIM_transformSingleNumericalValue_right_parameters():
    counterfactualInterfaceModel = StaticObjects.staticCounterfactualInterfaceModel()
    counterfactualInterfaceModel.openChosenDataset('German-Credit')
    counterfactualInterfaceModel.transformSingleNumericalValue('Age', 67)

# invertTransformedDataPoint(self, transformedDataPoint)
# the function invertTransformedDataPoint expect a not empty array
# send none as data point would arrise an assertionError
def test_CIM_invertTransformedDataPoint_none_parameter():
    with pytest.raises(AssertionError):
        counterfactualInterfaceModel = StaticObjects.staticCounterfactualInterfaceModel()
        counterfactualInterfaceModel.invertTransformedDataPoint(None)

# the function invertTransformedDataPoint expect a not empty array, and valid length
# send an invalid length as data point would arrise an assertionError
def test_CIM_invertTransformedDataPoint_wrong_length():
    with pytest.raises(AssertionError):
        counterfactualInterfaceModel = StaticObjects.staticCounterfactualInterfaceModel()
        counterfactualInterfaceModel.openChosenDataset('German-Credit')
        counterfactualInterfaceModel.transformDataPoint([0.14285714285714285, 1.0, 0.0, 0.6666666666666666, 0.09631891548633573, 0.0, 0.0, 1.0, -0.0, 1.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0])

# the function invertTransformedDataPoint expect a not empty array, and it with valid length
# would not arrise assertionError
def test_CIM_invertTransformedDataPoint_right_parameter():
    counterfactualInterfaceModel = StaticObjects.staticCounterfactualInterfaceModel()
    counterfactualInterfaceModel.openChosenDataset('German-Credit')
    counterfactualInterfaceModel.invertTransformedDataPoint([0.14285714285714285, 1.0, 0.0, 0.6666666666666666, 0.09631891548633573, 0.0, 0.0, 1.0, -0.0, 1.0, -0.0, 1.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0])
