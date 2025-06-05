using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Data;

public class BreastCancerWisconsinDiagnosticDataset : Dataset<
    BreastCancerWisconsinDiagnosticDataset.
    BreastCancerWisconsinDiagnosticModelInput>
{
    protected override string ResourceName { get; } =
        "Italbytz.ML.UCIMLR.Data.Breast_Cancer_Wisconsin_Diagnostic_.csv";

    public override string FilePrefix { get; } = "bcwd";

    public override string? LabelColumnName { get; } = @"Diagnosis";

    protected override string ColumnPropertiesString { get; } = """
        [
          {
            "ColumnName": "radius1",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "texture1",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "perimeter1",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "area1",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "smoothness1",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "compactness1",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "concavity1",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "concave_points1",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "symmetry1",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "fractal_dimension1",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "radius2",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "texture2",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "perimeter2",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "area2",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "smoothness2",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "compactness2",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "concavity2",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "concave_points2",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "symmetry2",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "fractal_dimension2",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "radius3",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "texture3",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "perimeter3",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "area3",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "smoothness3",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "compactness3",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "concavity3",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "concave_points3",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "symmetry3",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "fractal_dimension3",
            "ColumnPurpose": "Feature",
            "ColumnDataFormat": "Single",
            "IsCategorical": false,
            "Type": "Column",
            "Version": 5
          },
          {
            "ColumnName": "Diagnosis",
            "ColumnPurpose": "Label",
            "ColumnDataFormat": "String",
            "IsCategorical": true,
            "Type": "Column",
            "Version": 5
          }
        ]
        """;


    public override IEstimator<ITransformer> BuildPipeline(MLContext mlContext,
        ScenarioType scenarioType,
        IEstimator<ITransformer> estimator)
    {
        var pipeline = mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"radius1", @"radius1"),
                new InputOutputColumnPair(@"texture1", @"texture1"),
                new InputOutputColumnPair(@"perimeter1", @"perimeter1"),
                new InputOutputColumnPair(@"area1", @"area1"),
                new InputOutputColumnPair(@"smoothness1", @"smoothness1"),
                new InputOutputColumnPair(@"compactness1", @"compactness1"),
                new InputOutputColumnPair(@"concavity1", @"concavity1"),
                new InputOutputColumnPair(@"concave_points1",
                    @"concave_points1"),
                new InputOutputColumnPair(@"symmetry1", @"symmetry1"),
                new InputOutputColumnPair(@"fractal_dimension1",
                    @"fractal_dimension1"),
                new InputOutputColumnPair(@"radius2", @"radius2"),
                new InputOutputColumnPair(@"texture2", @"texture2"),
                new InputOutputColumnPair(@"perimeter2", @"perimeter2"),
                new InputOutputColumnPair(@"area2", @"area2"),
                new InputOutputColumnPair(@"smoothness2", @"smoothness2"),
                new InputOutputColumnPair(@"compactness2", @"compactness2"),
                new InputOutputColumnPair(@"concavity2", @"concavity2"),
                new InputOutputColumnPair(@"concave_points2",
                    @"concave_points2"),
                new InputOutputColumnPair(@"symmetry2", @"symmetry2"),
                new InputOutputColumnPair(@"fractal_dimension2",
                    @"fractal_dimension2"),
                new InputOutputColumnPair(@"radius3", @"radius3"),
                new InputOutputColumnPair(@"texture3", @"texture3"),
                new InputOutputColumnPair(@"perimeter3", @"perimeter3"),
                new InputOutputColumnPair(@"area3", @"area3"),
                new InputOutputColumnPair(@"smoothness3", @"smoothness3"),
                new InputOutputColumnPair(@"compactness3", @"compactness3"),
                new InputOutputColumnPair(@"concavity3", @"concavity3"),
                new InputOutputColumnPair(@"concave_points3",
                    @"concave_points3"),
                new InputOutputColumnPair(@"symmetry3", @"symmetry3"),
                new InputOutputColumnPair(@"fractal_dimension3",
                    @"fractal_dimension3")
            })
            .Append(mlContext.Transforms.Concatenate(@"Features", @"radius1",
                @"texture1", @"perimeter1", @"area1", @"smoothness1",
                @"compactness1", @"concavity1", @"concave_points1",
                @"symmetry1", @"fractal_dimension1", @"radius2", @"texture2",
                @"perimeter2", @"area2", @"smoothness2", @"compactness2",
                @"concavity2", @"concave_points2", @"symmetry2",
                @"fractal_dimension2", @"radius3", @"texture3", @"perimeter3",
                @"area3", @"smoothness3", @"compactness3", @"concavity3",
                @"concave_points3", @"symmetry3", @"fractal_dimension3"))
            .Append(mlContext.Transforms.Conversion.MapValueToKey(@"Diagnosis",
                @"Diagnosis", addKeyValueAnnotationsAsText: false))
            .Append(estimator)
            .Append(
                mlContext.Transforms.Conversion.MapKeyToValue(@"PredictedLabel",
                    @"PredictedLabel"));

        return pipeline;
    }

    public override IDataView LoadFromTextFile(string path,
        char separatorChar = IDataset.TextLoaderDefaults.Separator,
        bool hasHeader = IDataset.TextLoaderDefaults.HasHeader,
        bool allowQuoting = IDataset.TextLoaderDefaults.AllowQuoting,
        bool trimWhitespace = IDataset.TextLoaderDefaults.TrimWhitespace,
        bool allowSparse = IDataset.TextLoaderDefaults.AllowSparse)
    {
        return LoadFromTextFile<BreastCancerWisconsinDiagnosticModelInput>(path,
            separatorChar,
            hasHeader,
            allowQuoting, trimWhitespace, allowSparse);
    }

    public class BreastCancerWisconsinDiagnosticModelInput
    {
        [LoadColumn(0)]
        [ColumnName(@"radius1")]
        public float Radius1 { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"texture1")]
        public float Texture1 { get; set; }

        [LoadColumn(2)]
        [ColumnName(@"perimeter1")]
        public float Perimeter1 { get; set; }

        [LoadColumn(3)] [ColumnName(@"area1")] public float Area1 { get; set; }

        [LoadColumn(4)]
        [ColumnName(@"smoothness1")]
        public float Smoothness1 { get; set; }

        [LoadColumn(5)]
        [ColumnName(@"compactness1")]
        public float Compactness1 { get; set; }

        [LoadColumn(6)]
        [ColumnName(@"concavity1")]
        public float Concavity1 { get; set; }

        [LoadColumn(7)]
        [ColumnName(@"concave_points1")]
        public float Concave_points1 { get; set; }

        [LoadColumn(8)]
        [ColumnName(@"symmetry1")]
        public float Symmetry1 { get; set; }

        [LoadColumn(9)]
        [ColumnName(@"fractal_dimension1")]
        public float Fractal_dimension1 { get; set; }

        [LoadColumn(10)]
        [ColumnName(@"radius2")]
        public float Radius2 { get; set; }

        [LoadColumn(11)]
        [ColumnName(@"texture2")]
        public float Texture2 { get; set; }

        [LoadColumn(12)]
        [ColumnName(@"perimeter2")]
        public float Perimeter2 { get; set; }

        [LoadColumn(13)]
        [ColumnName(@"area2")]
        public float Area2 { get; set; }

        [LoadColumn(14)]
        [ColumnName(@"smoothness2")]
        public float Smoothness2 { get; set; }

        [LoadColumn(15)]
        [ColumnName(@"compactness2")]
        public float Compactness2 { get; set; }

        [LoadColumn(16)]
        [ColumnName(@"concavity2")]
        public float Concavity2 { get; set; }

        [LoadColumn(17)]
        [ColumnName(@"concave_points2")]
        public float Concave_points2 { get; set; }

        [LoadColumn(18)]
        [ColumnName(@"symmetry2")]
        public float Symmetry2 { get; set; }

        [LoadColumn(19)]
        [ColumnName(@"fractal_dimension2")]
        public float Fractal_dimension2 { get; set; }

        [LoadColumn(20)]
        [ColumnName(@"radius3")]
        public float Radius3 { get; set; }

        [LoadColumn(21)]
        [ColumnName(@"texture3")]
        public float Texture3 { get; set; }

        [LoadColumn(22)]
        [ColumnName(@"perimeter3")]
        public float Perimeter3 { get; set; }

        [LoadColumn(23)]
        [ColumnName(@"area3")]
        public float Area3 { get; set; }

        [LoadColumn(24)]
        [ColumnName(@"smoothness3")]
        public float Smoothness3 { get; set; }

        [LoadColumn(25)]
        [ColumnName(@"compactness3")]
        public float Compactness3 { get; set; }

        [LoadColumn(26)]
        [ColumnName(@"concavity3")]
        public float Concavity3 { get; set; }

        [LoadColumn(27)]
        [ColumnName(@"concave_points3")]
        public float Concave_points3 { get; set; }

        [LoadColumn(28)]
        [ColumnName(@"symmetry3")]
        public float Symmetry3 { get; set; }

        [LoadColumn(29)]
        [ColumnName(@"fractal_dimension3")]
        public float Fractal_dimension3 { get; set; }

        [LoadColumn(30)]
        [ColumnName(@"Diagnosis")]
        public string Diagnosis { get; set; }
    }
}