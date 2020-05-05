/**
 * Copyright or © or Copr. IETR/INSA - Rennes (2019 - 2020) :
 *
 * Karol Desnos <kdesnos@insa-rennes.fr> (2019 - 2020)
 *
 * GEGELATI is an open-source reinforcement learning framework for training
 * artificial intelligence based on Tangled Program Graphs (TPGs).
 *
 * This software is governed by the CeCILL-C license under French law and
 * abiding by the rules of distribution of free software. You can use,
 * modify and/ or redistribute the software under the terms of the CeCILL-C
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty and the software's author, the holder of the
 * economic rights, and the successive licensors have only limited
 * liability.
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading, using, modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean that it is complicated to manipulate, and that also
 * therefore means that it is reserved for developers and experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or
 * data to be ensured and, more generally, to use and operate it in the
 * same conditions as regards security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-C license and that you accept its terms.
 */

#ifndef DETERMINISTIC_RANDOM_H
#define DETERMINISTIC_RANDOM_H

#include <assert.h> 

#define _NODISCARD [[nodiscard]]

// For unsigned int 
namespace Mutator {

#ifndef DOXYGEN_SHOULD_SKIP_THIS

	// CLASS TEMPLATE _Rng_from_urng
	template <class _Diff, class _Urng>
	class _Rng_from_urng { // wrap a URNG as an RNG
	public:
		using _Ty0 = std::make_unsigned_t<_Diff>;
		using _Ty1 = typename _Urng::result_type;

		using _Udiff = std::conditional_t < sizeof(_Ty1) < sizeof(_Ty0), _Ty0, _Ty1 > ;

		explicit _Rng_from_urng(_Urng& _Func) : _Ref(_Func), _Bits(sizeof(char) * sizeof(_Udiff)), _Bmask(_Udiff(-1)) {
			for (; (_Urng::max)() - (_Urng::min)() < _Bmask; _Bmask >>= 1) {
				--_Bits;
			}
		}

		_Diff operator()(_Diff _Index) { // adapt _Urng closed range to [0, _Index)
			for (;;) { // try a sample random value
				_Udiff _Ret = 0; // random bits
				_Udiff _Mask = 0; // 2^N - 1, _Ret is within [0, _Mask]

				while (_Mask < _Udiff(_Index - 1)) { // need more random bits
					_Ret <<= _Bits - 1; // avoid full shift
					_Ret <<= 1;
					_Ret |= _Get_bits();
					_Mask <<= _Bits - 1; // avoid full shift
					_Mask <<= 1;
					_Mask |= _Bmask;
				}

				// _Ret is [0, _Mask], _Index - 1 <= _Mask, return if unbiased
				if (_Ret / _Index < _Mask / _Index || _Mask % _Index == _Udiff(_Index - 1)) {
					return static_cast<_Diff>(_Ret % _Index);
				}
			}
		}

		_Udiff _Get_all_bits() { // return a random value
			_Udiff _Ret = 0;

			for (size_t _Num = 0; _Num < sizeof(char) * sizeof(_Udiff); _Num += _Bits) { // don't mask away any bits
				_Ret <<= _Bits - 1; // avoid full shift
				_Ret <<= 1;
				_Ret |= _Get_bits();
			}

			return _Ret;
		}

		_Rng_from_urng(const _Rng_from_urng&) = delete;
		_Rng_from_urng& operator=(const _Rng_from_urng&) = delete;

	private:
		_Udiff _Get_bits() { // return a random value within [0, _Bmask]
			for (;;) { // repeat until random value is in range
				_Udiff _Val = _Ref() - (_Urng::min)();

				if (_Val <= _Bmask) {
					return _Val;
				}
			}
		}

		_Urng& _Ref; // reference to URNG
		size_t _Bits; // number of random bits generated by _Get_bits()
		_Udiff _Bmask; // 2^_Bits - 1
	};

	// CLASS TEMPLATE uniform_int
	template <class _Ty = int>
	class uniform_int { // uniform integer distribution
	public:
		using result_type = _Ty;

		struct param_type { // parameter package
			using distribution_type = uniform_int;

			explicit param_type(result_type _Min0 = 0, result_type _Max0 = 9) { // construct from parameters
				_Init(_Min0, _Max0);
			}

			_NODISCARD bool operator==(const param_type& _Right) const { // test for equality
				return _Min == _Right._Min && _Max == _Right._Max;
			}

			_NODISCARD bool operator!=(const param_type& _Right) const { // test for inequality
				return !(*this == _Right);
			}

			_NODISCARD result_type a() const { // return a value
				return _Min;
			}

			_NODISCARD result_type b() const { // return b value
				return _Max;
			}

			void _Init(_Ty _Min0, _Ty _Max0) { // set internal state
				assert(_Min0 <= _Max0);//, "invalid min and max arguments for uniform_int");
				_Min = _Min0;
				_Max = _Max0;
			}

			result_type _Min;
			result_type _Max;
		};

		explicit uniform_int(_Ty _Min0 = 0, _Ty _Max0 = 9) : _Par(_Min0, _Max0) { // construct from parameters
		}

		explicit uniform_int(const param_type& _Par0) : _Par(_Par0) { // construct from parameter package
		}

		_NODISCARD result_type a() const { // return a value
			return _Par.a();
		}

		_NODISCARD result_type b() const { // return b value
			return _Par.b();
		}

		_NODISCARD param_type param() const { // return parameter package
			return _Par;
		}

		void param(const param_type& _Par0) { // set parameter package
			_Par = _Par0;
		}

		_NODISCARD result_type(min)() const { // return minimum possible generated value
			return _Par._Min;
		}

		_NODISCARD result_type(max)() const { // return maximum possible generated value
			return _Par._Max;
		}

		void reset() { // clear internal state
		}

		template <class _Engine>
		_NODISCARD result_type operator()(_Engine& _Eng) const { // return next value
			return _Eval(_Eng, _Par._Min, _Par._Max);
		}

		template <class _Engine>
		_NODISCARD result_type operator()(
			_Engine& _Eng, const param_type& _Par0) const { // return next value, given parameter package
			return _Eval(_Eng, _Par0._Min, _Par0._Max);
		}

		template <class _Engine>
		_NODISCARD result_type operator()(_Engine& _Eng, result_type _Nx) const { // return next value
			return _Eval(_Eng, 0, _Nx - 1);
		}

		template <class _Elem, class _Traits>
		std::basic_istream<_Elem, _Traits>& _Read(std::basic_istream<_Elem, _Traits>& _Istr) { // read state from _Istr
			_Ty _Min0;
			_Ty _Max0;
			_Istr >> _Min0 >> _Max0;
			_Par._Init(_Min0, _Max0);
			return _Istr;
		}

		template <class _Elem, class _Traits>
		std::basic_ostream<_Elem, _Traits>& _Write(std::basic_ostream<_Elem, _Traits>& _Ostr) const { // write state to _Ostr
			return _Ostr << _Par._Min << ' ' << _Par._Max;
		}

	private:
		using _Uty = std::make_unsigned_t<_Ty>;

		template <class _Engine>
		result_type _Eval(_Engine& _Eng, _Ty _Min, _Ty _Max) const { // compute next value in range [_Min, _Max]
			_Rng_from_urng<_Uty, _Engine> _Rng(_Eng);

			const _Uty _Umin = _Adjust(_Uty(_Min));
			const _Uty _Umax = _Adjust(_Uty(_Max));

			_Uty _Uret;

			if (_Umax - _Umin == _Uty(-1)) {
				_Uret = static_cast<_Uty>(_Rng._Get_all_bits());
			}
			else {
				_Uret = static_cast<_Uty>(_Rng(static_cast<_Uty>(_Umax - _Umin + 1)));
			}

			return _Ty(_Adjust(static_cast<_Uty>(_Uret + _Umin)));
		}

		static _Uty _Adjust(_Uty _Uval) { // convert signed ranges to unsigned ranges and vice versa
			return _Adjust(_Uval, std::is_signed<_Ty>());
		}

		static _Uty _Adjust(_Uty _Uval, std::true_type) { // convert signed ranges to unsigned ranges and vice versa
			const _Uty _Adjuster = (_Uty(-1) >> 1) + 1; // 2^(N-1)

			if (_Uval < _Adjuster) {
				return static_cast<_Uty>(_Uval + _Adjuster);
			}
			else {
				return static_cast<_Uty>(_Uval - _Adjuster);
			}
		}

		static _Uty _Adjust(_Uty _Uval, std::false_type) { // _Ty is already unsigned, do nothing
			return _Uval;
		}

		param_type _Par;
	};

	template <class _Elem, class _Traits, class _Ty>
	std::basic_istream<_Elem, _Traits>& operator>>(std::basic_istream<_Elem, _Traits>& _Istr,
		uniform_int<_Ty>& _Dist) { // read state from _Istr
		return _Dist._Read(_Istr);
	}

	template <class _Elem, class _Traits, class _Ty>
	std::basic_ostream<_Elem, _Traits>& operator<<(std::basic_ostream<_Elem, _Traits>& _Ostr,
		const uniform_int<_Ty>& _Dist) { // write state to _Ostr
		return _Dist._Write(_Ostr);
	}

	// CLASS TEMPLATE uniform_int_distribution
	template <class _Ty = int>
	class uniform_int_distribution : public uniform_int<_Ty> { // uniform integer distribution
	public:
		// _RNG_REQUIRE_INTTYPE(uniform_int_distribution, _Ty); // From visual Removed

		using _Mybase = uniform_int<_Ty>;
		using _Mypbase = typename _Mybase::param_type;
		using result_type = typename _Mybase::result_type;

		struct param_type : public _Mypbase { // parameter package
			using distribution_type = uniform_int_distribution;

			explicit param_type(result_type _Min0 = 0, result_type _Max0 = (std::numeric_limits<_Ty>::max)())
				: _Mypbase(_Min0, _Max0) { // construct from parameters
			}

			param_type(const _Mypbase& _Right) : _Mypbase(_Right) { // construct from base
			}
		};

		explicit uniform_int_distribution(_Ty _Min0 = 0, _Ty _Max0 = (std::numeric_limits<_Ty>::max)())
			: _Mybase(_Min0, _Max0) { // construct from parameters
		}

		explicit uniform_int_distribution(const param_type& _Par0) : _Mybase(_Par0) { // construct from parameter package
		}
	};

	template <class _Ty>
	_NODISCARD bool operator==(const uniform_int_distribution<_Ty>& _Left,
		const uniform_int_distribution<_Ty>& _Right) { // test for equality
		return _Left.param() == _Right.param();
	}

	template <class _Ty>
	_NODISCARD bool operator!=(const uniform_int_distribution<_Ty>& _Left,
		const uniform_int_distribution<_Ty>& _Right) { // test for inequality
		return !(_Left == _Right);
	}
#endif
}
#include <istream>
// For Double
namespace Mutator {
#ifndef DOXYGEN_SHOULD_SKIP_THIS

#define _NRAND(eng, resty) (std::generate_canonical<resty, static_cast<size_t>(-1)>(eng))

	// CLASS TEMPLATE uniform_real
	template <class _Ty = double>
	class uniform_real { // uniform real distribution
	public:
		using result_type = _Ty;

		struct param_type { // parameter package
			using distribution_type = uniform_real;

			explicit param_type(_Ty _Min0 = _Ty{ 0 }, _Ty _Max0 = _Ty{ 1 }) {
				_Init(_Min0, _Max0);
			}

			_NODISCARD bool operator==(const param_type& _Right) const {
				return _Min == _Right._Min && _Max == _Right._Max;
			}

			_NODISCARD bool operator!=(const param_type& _Right) const {
				return !(*this == _Right);
			}

			_NODISCARD result_type a() const {
				return _Min;
			}

			_NODISCARD result_type b() const {
				return _Max;
			}

			void _Init(_Ty _Min0, _Ty _Max0) { // set internal state
				// From Visual : removed
				// _STL_ASSERT(_Min0 <= _Max0 && (0 <= _Min0 || _Max0 <= _Min0 + (std::numeric_limits<_Ty>::max)()),
				//	"invalid min and max arguments for uniform_real");
				_Min = _Min0;
				_Max = _Max0;
			}

			result_type _Min;
			result_type _Max;
		};

		explicit uniform_real(_Ty _Min0 = _Ty{ 0 }, _Ty _Max0 = _Ty{ 1 }) : _Par(_Min0, _Max0) {}

		explicit uniform_real(const param_type& _Par0) : _Par(_Par0) {}

		_NODISCARD result_type a() const {
			return _Par.a();
		}

		_NODISCARD result_type b() const {
			return _Par.b();
		}

		_NODISCARD param_type param() const {
			return _Par;
		}

		void param(const param_type& _Par0) { // set parameter package
			_Par = _Par0;
		}

		_NODISCARD result_type(min)() const {
			return _Par._Min;
		}

		_NODISCARD result_type(max)() const {
			return _Par._Max;
		}

		void reset() { // clear internal state
		}

		template <class _Engine>
		_NODISCARD result_type operator()(_Engine& _Eng) const {
			return _Eval(_Eng, _Par);
		}

		template <class _Engine>
		_NODISCARD result_type operator()(_Engine& _Eng, const param_type& _Par0) const {
			return _Eval(_Eng, _Par0);
		}

		template <class _Elem, class _Traits>
		std::basic_istream<_Elem, _Traits>& _Read(std::basic_istream<_Elem, _Traits>& _Istr) { // read state from _Istr
			_Ty _Min0;
			_Ty _Max0;
			_In(_Istr, _Min0);
			_In(_Istr, _Max0);
			_Par._Init(_Min0, _Max0);
			return _Istr;
		}

		template <class _Elem, class _Traits>
		std::basic_ostream<_Elem, _Traits>& _Write(std::basic_ostream<_Elem, _Traits>& _Ostr) const { // write state to _Ostr
			_Out(_Ostr, _Par._Min);
			_Out(_Ostr, _Par._Max);
			return _Ostr;
		}

	private:
		template <class _Engine>
		result_type _Eval(_Engine& _Eng, const param_type& _Par0) const {
			return _NRAND(_Eng, _Ty) * (_Par0._Max - _Par0._Min) + _Par0._Min;
		}

		param_type _Par;
	};

	template <class _Elem, class _Traits, class _Ty>
	std::basic_istream<_Elem, _Traits>& operator>>(std::basic_istream<_Elem, _Traits>& _Istr,
		uniform_real<_Ty>& _Dist) { // read state from _Istr
		return _Dist._Read(_Istr);
	}

	template <class _Elem, class _Traits, class _Ty>
	std::basic_ostream<_Elem, _Traits>& operator<<(std::basic_ostream<_Elem, _Traits>& _Ostr,
		const uniform_real<_Ty>& _Dist) { // write state to _Ostr
		return _Dist._Write(_Ostr);
	}


	// CLASS TEMPLATE uniform_real_distribution
	template <class _Ty = double>
	class uniform_real_distribution : public uniform_real<_Ty> { // uniform real distribution
	public:
		//_RNG_REQUIRE_REALTYPE(uniform_real_distribution, _Ty);

		using _Mybase = uniform_real<_Ty>;
		using _Mypbase = typename _Mybase::param_type;
		using result_type = typename _Mybase::result_type;

		struct param_type : public _Mypbase { // parameter package
			using distribution_type = uniform_real_distribution;

			explicit param_type(_Ty _Min0 = _Ty{ 0 }, _Ty _Max0 = _Ty{ 1 }) : _Mypbase(_Min0, _Max0) {}

			param_type(const _Mypbase& _Right) : _Mypbase(_Right) {}
		};

		explicit uniform_real_distribution(_Ty _Min0 = _Ty{ 0 }, _Ty _Max0 = _Ty{ 1 }) : _Mybase(_Min0, _Max0) {}

		explicit uniform_real_distribution(const param_type& _Par0) : _Mybase(_Par0) {}
	};

	template <class _Ty>
	_NODISCARD bool operator==(const uniform_real_distribution<_Ty>& _Left, const uniform_real_distribution<_Ty>& _Right) {
		return _Left.param() == _Right.param();
	}

	template <class _Ty>
	_NODISCARD bool operator!=(const uniform_real_distribution<_Ty>& _Left, const uniform_real_distribution<_Ty>& _Right) {
		return !(_Left == _Right);
	}
#endif
}

#endif 

