using Microsoft.ML;

namespace Italbytz.ML.UCIMLR;

public class IrisDataset : Dataset
{
    protected override string ResourceName { get; } =
        "Italbytz.ML.UCIMLR.Data.Iris.csv";

    protected override string FilePrefix { get; } = "iris";

    protected override string ColumnPropertiesString { get; } = """
        [
          {
            "ColumnName": "sepal length",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "sepal width",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "petal length",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "petal width",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "class",
            "ColumnPurpose": "Label",
            "ColumnDataFormat": "String",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          }
        ]
        """;

    protected override IDataView? LoadFromTextFile(string tempFile)
    {
        var mlContext = new MLContext();

        // Load the dataset from the temporary file
        var data = mlContext.Data.LoadFromTextFile<IrisModelInput>(
            tempFile, ',', true);

        return data;
    }
}