#include <gtest/gtest.h>

#include "data/primitiveTypeArray2D.h"

TEST(PrimitiveTypeArray2DTest, Constructor)
{
    Data::PrimitiveTypeArray2D<double>* array;
    ASSERT_NE(array = new Data::PrimitiveTypeArray2D<double>(3, 4), nullptr)
        << "A PrimitiveTypeArray2D<double> could not be built successfully.";
    ASSERT_NO_THROW(delete array)
        << "PrimitiveTypeArray2D could not be deleted.";
}

TEST(PrimitiveTypeArray2DTest, getAddressSpace)
{
    size_t h = 3;
    size_t w = 5;
    Data::PrimitiveTypeArray2D<int> a(w, h);

    // Check primitive type provided by 1D array
    ASSERT_EQ(a.getAddressSpace(typeid(int)), w * h)
        << "Address space of the 2D array of int is not width*height for "
           "typeid(int).";

    ASSERT_EQ(a.getAddressSpace(typeid(int[2])), (w - 2 + 1) * h)
        << "Address space of the 2D array of int is not correct for "
           "typeid(int[2]).";

    // Request a 2D array with valid dimensions
    ASSERT_EQ(a.getAddressSpace(typeid(int[2][4])), (w - 4 + 1) * (h - 2 + 1))
        << "Returned address space for int[2][4] in a 2D int array of size 5x3 "
           "is incorrect.";

    // Request a const 2D array with valid dimensions
    ASSERT_EQ(a.getAddressSpace(typeid(const int[2][4])),
              (w - 4 + 1) * (h - 2 + 1))
        << "Returned address space for int[2][4] in a 2D int array of size 5x3 "
           "is incorrect.";

    // Request a 2D array with invalid dimensions
    ASSERT_EQ(a.getAddressSpace(typeid(int[4][2])), 0)
        << "Returned address space for int[4][2] in a 2D int array of size 5x3 "
           "is incorrect.";

    // Request a 2D array with invalid type
    ASSERT_EQ(a.getAddressSpace(typeid(long[1][1])), 0)
        << "Returned address space for int[4][2] in a 2D int array of size 5x3 "
           "is incorrect.";
}

TEST(PrimitiveTypeArray2DTest, getDataAt)
{
    const size_t h = 3;
    const size_t w = 5;
    Data::PrimitiveTypeArray2D<int> a(w, h);

    // Fill the array
    for (auto idx = 0; idx < h * w; idx++) {
        a.setDataAt(typeid(int), idx, idx);
    }

    // Check primitive type provided by 1D array
    for (auto idx = 0; idx < h * w; idx++) {
        const int val =
            *((a.getDataAt(typeid(int), idx)).getSharedPointer<const int>());
        ASSERT_EQ(val, idx) << "Value with primitive type is not as expected.";
    }

    // Check 1D array
    for (auto idx = 0; idx < a.getAddressSpace(typeid(int[3])); idx++) {
        std::shared_ptr<const int> valSPtr =
            (a.getDataAt(typeid(int[3]), idx)).getSharedPointer<const int[]>();
        const int* valPtr = valSPtr.get();
        for (auto subIdx = 0; subIdx < 3; subIdx++) {
            const int val = valPtr[subIdx];
            ASSERT_EQ(val, (idx / (w - 3 + 1) * w + idx % (w - 3 + 1)) + subIdx)
                << "Value with primitive type is not as expected.";
        }
    }

    // Check 2D array (returned as a 1D array)
    for (auto idx = 0; idx < a.getAddressSpace(typeid(int[3][2])); idx++) {
        std::shared_ptr<const int> valSPtr =
            (a.getDataAt(typeid(int[2][3]), idx))
                .getSharedPointer<const int[]>();
        const int(*valPtr)[3] = (int(*)[3])valSPtr.get();
        size_t srcIdx = idx / (w - 3 + 1) * w + idx % (w - 3 + 1);
        for (auto subH = 0; subH < 2; subH++) {
            for (auto subW = 0; subW < 3; subW++) {
                const int val = valPtr[subH][subW];
                ASSERT_EQ(val, srcIdx + (subH * w) + subW)
                    << "Value with primitive type is not as expected.";
            }
        }
    }

#ifndef NDEBUG
    ASSERT_THROW(a.getDataAt(typeid(int[h * w]), 1), std::invalid_argument)
        << "Address exceeding the addressSpace should cause an exception.";

    ASSERT_THROW(a.getDataAt(typeid(int[w - 1]), h * (w - 1) + 1),
                 std::out_of_range)
        << "Address exceeding the addressSpace should cause an exception.";
#else
    // No alternative test to put here.. out of range access to memory _may_
    // happen without being detected.
#endif
}