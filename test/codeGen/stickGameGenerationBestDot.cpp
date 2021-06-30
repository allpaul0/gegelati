#include <file/parametersParser.h>
#include <gtest/gtest.h>
#include <stddef.h>

#include "cfloat"
#include "code_gen/LambdaPrintableInstruction.h"
#include "code_gen/ProgramGenerationEngine.h"
#include "code_gen/TpgGenerationEngine.h"
#include "environment.h"
#include "file/tpgGraphDotImporter.h"
#include "instructions/set.h"
#include "tpg/tpgAction.h"
#include "tpg/tpgGraph.h"
#include "tpg/tpgTeam.h"
#include "tpg/tpgVertex.h"
#include <file/parametersParser.h>
#include <iostream>

class StickGameGenerationBestDotTest : public ::testing::Test
{
  protected:
    const size_t sizeHint{4};
    const size_t sizeRemainingStick{1};
    Instructions::Set set;
    Environment* e;
    std::vector<std::reference_wrapper<const Data::DataHandler>> data;
    Data::PrimitiveTypeArray<int> hints{sizeHint};
    Data::PrimitiveTypeArray<int> remainingSticks{sizeRemainingStick};
    TPG::TPGGraph* tpg;
    CodeGen::TPGGenerationEngine* tpgGen;
    File::TPGGraphDotImporter* dot = nullptr;
    std::string cmdCompile{"make -C " TESTS_DAT_PATH "/codeGen"};

    virtual void SetUp()
    {

        auto minus = [](int a, int b) -> double {
            return (double)a - (double)b;
        };
        auto cast = [](int a) -> double { return (double)a; };
        auto add = [](double a, double b) -> double { return a + b; };
        auto max = [](double a, double b) -> double { return std::max(a, b); };
        auto nulltest = [](double a) -> double {
            return (a == 0.0) ? 10.0 : 0.0;
        };
        auto modulo = [](double a, double b) -> double {
            if (b != 0.0) {
                return fmod(a, b);
            }
            else {
                return DBL_MIN;
            }
        };

        set.add(*(new Instructions::LambdaPrintableInstruction<double, double>(
            "$0 = (($2) != 0.0) ? fmod($1, $2) : DBL_MIN ;", modulo)));
        set.add(*(new Instructions::LambdaPrintableInstruction<int, int>(
            "$0 = (double)($1) - (double)($2);", minus)));
        set.add(*(new Instructions::LambdaPrintableInstruction<double, double>(
            "$0 = $1 + $2;", add)));
        set.add(*(new Instructions::LambdaPrintableInstruction<int>(
            "$0 = (double)($1);", cast)));
        set.add(*(new Instructions::LambdaPrintableInstruction<double, double>(
            "$0 = (($1) < ($2)) ? ($2) : ($1); ", max)));
        set.add(*(new Instructions::LambdaPrintableInstruction<double>(
            "$0 = ($1 == 0.0) ? 10.0 : 0.0;", nulltest)));
        data = {hints, remainingSticks};

        e = new Environment(set, data, 8);
        tpg = new TPG::TPGGraph(*e);
    }

    virtual void TearDown()
    {

        delete e;
        delete tpg;

        delete (&set.getInstruction(0));
        delete (&set.getInstruction(1));
        delete (&set.getInstruction(2));
        delete (&set.getInstruction(3));
        delete (&set.getInstruction(4));
        delete (&set.getInstruction(5));

        delete dot;
    }
};

TEST_F(StickGameGenerationBestDotTest, BestTPG)
{
    dot = new File::TPGGraphDotImporter(TESTS_DAT_PATH "StickGame_out_best.dot",
                                        *e, *tpg);
    ASSERT_NO_THROW(dot->importGraph())
        << "Failed to Import the graph to test inference of stick game";
    tpgGen =
        new CodeGen::TPGGenerationEngine("StickGameBest_TPG", *tpg, "../src/");
    ASSERT_NO_THROW(tpgGen->generateTPGGraph())
        << "Fail to generate the C file to test StickGame";
    // call destructor to close generated files
    delete tpgGen;

    cmdCompile += " StickGameBest_TPG";

    ASSERT_EQ(system(cmdCompile.c_str()), 0)
        << "Fail to compile generated files to test stick game";
    int result = WEXITSTATUS(system("./StickGameBest_TPG"));

    ASSERT_EQ(result, 2)
        << "Error inference of Stick Game has changed";
}
