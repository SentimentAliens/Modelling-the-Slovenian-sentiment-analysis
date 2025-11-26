# Modelling-the-Slovenian-sentiment-analysis
## Research Questions
- What is the number of misllabelled number of instances present in the given dataset ?
- How much performance can be improved when a cleaner dataset is used for training ?
  
## Links
- [DATASET](https://www.clarin.si/repository/xmlui/handle/11356/1115)
- [STEPS](/codes/STEPS.md)

## Methodology
<ol>
  <li>Detection of erronous labels in dataset. Refer STEPS.md for pseudo code and explanation</li>
  <li>Manual Correction of those labels</li>
  <li>Using the test set to get zero-shot accuracy on GAMS model TODO: choose a model that can fit in 40GB GPU </li>
  <li>Training</li>
  <li>Evalution</li>
  <ol>
      <li>Performance metric : Precision, Recall, F1, Accuracy</li>
      <li>Qualitative analysis</li>
    </ol>
</ol>
 
 
