using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Italbytz.ML.Data.Tests.Unit;

[TestClass]
public class SplittingTests
{
    [TestMethod]
    public async Task SplitAndLoadTest()
    {
        ThreadSafeMLContext.Seed = 42;
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var dataset = Data.NPHA;
        var splitRatio = 0.2f;
        var tmpDir = Path.GetTempPath();
        var files =
            dataset.GetTrainValidateTestFiles(tmpDir,
                validateFraction: splitRatio, testFraction: splitRatio,
                seeds: [42]).ToList();
        var dataView =
            dataset.LoadFromTextFile(Path.Combine(tmpDir,
                files[0].TrainValidateFileName), ',', true);
        //var dataView = dataset.DataView;
        var pipeline = dataset.BuildPreprocessingPipeline(mlContext);
        var model = pipeline.Fit(dataView);
        var result = model.Transform(dataView);

        var excerpt = result.GetDataExcerpt();
        var features = excerpt.Features;
        var labels = excerpt.Labels;
    }
}