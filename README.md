# Modelling-the-Slovenian-sentiment-analysis
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
 
 
