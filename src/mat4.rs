use crate::core::{
    storage::{Vector4x4, XYZW},
    traits::{
        matrix::{FloatMatrix4x4, Matrix4x4, MatrixConst},
        projection::ProjectionMatrix,
    },
};
use crate::{DQuat, DVec3, DVec4, Quat, Vec3, Vec3A, Vec4};

#[cfg(all(
    target_feature = "sse2",
    not(feature = "scalar-math"),
    target_arch = "x86"
))]
use core::arch::x86::*;
#[cfg(all(
    target_feature = "sse2",
    not(feature = "scalar-math"),
    target_arch = "x86_64"
))]
use core::arch::x86_64::*;

#[cfg(not(target_arch = "spirv"))]
use core::fmt;
use core::{
    cmp::Ordering,
    ops::{Add, Deref, DerefMut, Mul, Sub},
};

#[cfg(feature = "std")]
use std::iter::{Product, Sum};

macro_rules! impl_mat4_methods {
    ($t:ty, $vec4:ident, $vec3:ident, $quat:ident, $inner:ident) => {
        /// A 4x4 matrix with all elements set to `0.0`.
        pub const ZERO: Self = Self($inner::ZERO);

        /// A 4x4 identity matrix, where all diagonal elements are `1`, and all off-diagonal elements are `0`.
        pub const IDENTITY: Self = Self($inner::IDENTITY);

        /// Creates a 4x4 matrix with all elements set to `0.0`.
        #[deprecated = "use Mat4::ZERO instead"]
        #[inline(always)]
        pub const fn zero() -> Self {
            Self::ZERO
        }

        /// Creates a 4x4 identity matrix.
        #[deprecated = "use Mat4::IDENTITY instead"]
        #[inline(always)]
        pub const fn identity() -> Self {
            Self::IDENTITY
        }

        /// Creates a 4x4 matrix from four column vectors.
        #[inline(always)]
        pub fn from_cols(x_axis: $vec4, y_axis: $vec4, z_axis: $vec4, w_axis: $vec4) -> Self {
            Self($inner::from_cols(x_axis.0, y_axis.0, z_axis.0, w_axis.0))
        }

        /// Creates a 4x4 matrix from a `[S; 16]` array stored in column major order.
        /// If your data is stored in row major you will need to `transpose` the returned
        /// matrix.
        #[inline(always)]
        pub fn from_cols_array(m: &[$t; 16]) -> Self {
            Self($inner::from_cols_array(m))
        }

        /// Creates a `[S; 16]` array storing data in column major order.
        /// If you require data in row major order `transpose` the matrix first.
        #[inline(always)]
        pub fn to_cols_array(&self) -> [$t; 16] {
            *self.as_ref()
        }

        /// Creates a 4x4 matrix from a `[[S; 4]; 4]` 2D array stored in column major order.
        /// If your data is in row major order you will need to `transpose` the returned
        /// matrix.
        #[inline(always)]
        pub fn from_cols_array_2d(m: &[[$t; 4]; 4]) -> Self {
            Self($inner::from_cols_array_2d(m))
        }

        /// Creates a `[[S; 4]; 4]` 2D array storing data in column major order.
        /// If you require data in row major order `transpose` the matrix first.
        #[inline(always)]
        pub fn to_cols_array_2d(&self) -> [[$t; 4]; 4] {
            self.0.to_cols_array_2d()
        }

        /// Creates a 4x4 matrix with its diagonal set to `diagonal` and all other entries set to 0.
        #[cfg_attr(docsrs, doc(alias = "scale"))]
        #[inline(always)]
        pub fn from_diagonal(diagonal: $vec4) -> Self {
            Self($inner::from_diagonal(diagonal.0))
        }

        /// Creates a 3D affine transformation matrix from the given `scale`, `rotation` and
        /// `translation`.
        ///
        /// The resulting matrix can be used to transform 3D points and vectors. See
        /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
        #[inline(always)]
        pub fn from_scale_rotation_translation(
            scale: $vec3,
            rotation: $quat,
            translation: $vec3,
        ) -> Self {
            Self($inner::from_scale_quaternion_translation(
                scale.0,
                rotation.0,
                translation.0,
            ))
        }

        /// Creates a 3D affine transformation matrix from the given `translation`.
        ///
        /// The resulting matrix can be used to transform 3D points and vectors. See
        /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
        #[inline(always)]
        pub fn from_rotation_translation(rotation: $quat, translation: $vec3) -> Self {
            Self($inner::from_quaternion_translation(
                rotation.0,
                translation.0,
            ))
        }

        /// Extracts `scale`, `rotation` and `translation` from `self`. The input matrix is
        /// expected to be a 3D affine transformation matrix otherwise the output will be invalid.
        #[inline(always)]
        pub fn to_scale_rotation_translation(&self) -> ($vec3, $quat, $vec3) {
            let (scale, rotation, translation) = self.0.to_scale_quaternion_translation();
            ($vec3(scale), $quat(rotation), $vec3(translation))
        }

        /// Creates a 3D affine transformation matrix from the given `rotation` quaternion.
        ///
        /// The resulting matrix can be used to transform 3D points and vectors. See
        /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
        #[inline(always)]
        pub fn from_quat(rotation: $quat) -> Self {
            Self($inner::from_quaternion(rotation.0))
        }

        /// Creates a 3D affine transformation matrix from the given `translation`.
        ///
        /// The resulting matrix can be used to transform 3D points and vectors. See
        /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
        #[inline(always)]
        pub fn from_translation(translation: $vec3) -> Self {
            Self($inner::from_translation(translation.0))
        }

        /// Creates a 3D affine transformation matrix containing a rotation around a normalized
        /// rotation `axis` of `angle` (in radians).
        ///
        /// The resulting matrix can be used to transform 3D points and vectors. See
        /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
        #[inline(always)]
        pub fn from_axis_angle(axis: $vec3, angle: $t) -> Self {
            Self($inner::from_axis_angle(axis.0, angle))
        }

        /// Creates a 3D affine transformation matrix containing a rotation around the given Euler
        /// angles (in radians).
        ///
        /// The resulting matrix can be used to transform 3D points and vectors. See
        /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
        #[inline(always)]
        pub fn from_rotation_ypr(yaw: $t, pitch: $t, roll: $t) -> Self {
            let quat = $quat::from_rotation_ypr(yaw, pitch, roll);
            Self::from_quat(quat)
        }

        /// Creates a 3D affine transformation matrix containing a rotation around the x axis of
        /// `angle` (in radians).
        ///
        /// The resulting matrix can be used to transform 3D points and vectors. See
        /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
        #[inline(always)]
        pub fn from_rotation_x(angle: $t) -> Self {
            Self($inner::from_rotation_x(angle))
        }

        /// Creates a 3D affine transformation matrix containing a rotation around the y axis of
        /// `angle` (in radians).
        ///
        /// The resulting matrix can be used to transform 3D points and vectors. See
        /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
        #[inline(always)]
        pub fn from_rotation_y(angle: $t) -> Self {
            Self($inner::from_rotation_y(angle))
        }

        /// Creates a 3D affine transformation matrix containing a rotation around the z axis of
        /// `angle` (in radians).
        ///
        /// The resulting matrix can be used to transform 3D points and vectors. See
        /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
        #[inline(always)]
        pub fn from_rotation_z(angle: $t) -> Self {
            Self($inner::from_rotation_z(angle))
        }

        /// Creates a 3D affine transformation matrix containing the given non-uniform `scale`.
        ///
        /// The resulting matrix can be used to transform 3D points and vectors. See
        /// [`Self::transform_point3()`] and [`Self::transform_vector3()`].
        #[inline(always)]
        pub fn from_scale(scale: $vec3) -> Self {
            Self($inner::from_scale(scale.0))
        }

        // #[inline]
        // pub(crate) fn col(&self, index: usize) -> $vec4 {
        //     match index {
        //         0 => self.x_axis,
        //         1 => self.y_axis,
        //         2 => self.z_axis,
        //         3 => self.w_axis,
        //         _ => panic!(
        //             "index out of bounds: the len is 4 but the index is {}",
        //             index
        //         ),
        //     }
        // }

        // #[inline]
        // pub(crate) fn col_mut(&mut self, index: usize) -> &mut $vec4 {
        //     match index {
        //         0 => &mut self.x_axis,
        //         1 => &mut self.y_axis,
        //         2 => &mut self.z_axis,
        //         3 => &mut self.w_axis,
        //         _ => panic!(
        //             "index out of bounds: the len is 4 but the index is {}",
        //             index
        //         ),
        //     }
        // }

        /// Returns `true` if, and only if, all elements are finite.
        /// If any element is either `NaN`, positive or negative infinity, this will return `false`.
        #[inline]
        pub fn is_finite(&self) -> bool {
            self.x_axis.is_finite()
                && self.y_axis.is_finite()
                && self.z_axis.is_finite()
                && self.w_axis.is_finite()
        }

        /// Returns `true` if any elements are `NaN`.
        #[inline]
        pub fn is_nan(&self) -> bool {
            self.x_axis.is_nan()
                || self.y_axis.is_nan()
                || self.z_axis.is_nan()
                || self.w_axis.is_nan()
        }

        /// Returns the transpose of `self`.
        #[inline(always)]
        pub fn transpose(&self) -> Self {
            Self(self.0.transpose())
        }

        /// Returns the determinant of `self`.
        #[inline(always)]
        pub fn determinant(&self) -> $t {
            self.0.determinant()
        }

        /// Returns the inverse of `self`.
        ///
        /// If the matrix is not invertible the returned matrix will be invalid.
        #[inline(always)]
        pub fn inverse(&self) -> Self {
            Self(self.0.inverse())
        }

        /// Creates a left-handed view matrix using a camera position, an up direction, and a focal
        /// point.
        #[inline(always)]
        pub fn look_at_lh(eye: $vec3, center: $vec3, up: $vec3) -> Self {
            Self($inner::look_at_lh(eye.0, center.0, up.0))
        }

        /// Creates a right-handed view matrix using a camera position, an up direction, and a focal
        /// point.
        #[inline(always)]
        pub fn look_at_rh(eye: $vec3, center: $vec3, up: $vec3) -> Self {
            Self($inner::look_at_rh(eye.0, center.0, up.0))
        }

        /// Creates a right-handed perspective projection matrix with [-1,1] depth range.
        /// This is the same as the OpenGL `gluPerspective` function.
        /// See https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml
        #[inline(always)]
        pub fn perspective_rh_gl(
            fov_y_radians: $t,
            aspect_ratio: $t,
            z_near: $t,
            z_far: $t,
        ) -> Self {
            Self($inner::perspective_rh_gl(
                fov_y_radians,
                aspect_ratio,
                z_near,
                z_far,
            ))
        }

        /// Creates a left-handed perspective projection matrix with [0,1] depth range.
        #[inline(always)]
        pub fn perspective_lh(fov_y_radians: $t, aspect_ratio: $t, z_near: $t, z_far: $t) -> Self {
            Self($inner::perspective_lh(
                fov_y_radians,
                aspect_ratio,
                z_near,
                z_far,
            ))
        }

        /// Creates a right-handed perspective projection matrix with [0,1] depth range.
        #[inline(always)]
        pub fn perspective_rh(fov_y_radians: $t, aspect_ratio: $t, z_near: $t, z_far: $t) -> Self {
            Self($inner::perspective_rh(
                fov_y_radians,
                aspect_ratio,
                z_near,
                z_far,
            ))
        }

        /// Creates an infinite left-handed perspective projection matrix with [0,1] depth range.
        #[inline(always)]
        pub fn perspective_infinite_lh(fov_y_radians: $t, aspect_ratio: $t, z_near: $t) -> Self {
            Self($inner::perspective_infinite_lh(
                fov_y_radians,
                aspect_ratio,
                z_near,
            ))
        }

        /// Creates an infinite left-handed perspective projection matrix with [0,1] depth range.
        #[inline(always)]
        pub fn perspective_infinite_reverse_lh(
            fov_y_radians: $t,
            aspect_ratio: $t,
            z_near: $t,
        ) -> Self {
            Self($inner::perspective_infinite_reverse_lh(
                fov_y_radians,
                aspect_ratio,
                z_near,
            ))
        }

        /// Creates an infinite right-handed perspective projection matrix with
        /// [0,1] depth range.
        #[inline(always)]
        pub fn perspective_infinite_rh(fov_y_radians: $t, aspect_ratio: $t, z_near: $t) -> Self {
            Self($inner::perspective_infinite_rh(
                fov_y_radians,
                aspect_ratio,
                z_near,
            ))
        }

        /// Creates an infinite reverse right-handed perspective projection matrix
        /// with [0,1] depth range.
        #[inline(always)]
        pub fn perspective_infinite_reverse_rh(
            fov_y_radians: $t,
            aspect_ratio: $t,
            z_near: $t,
        ) -> Self {
            Self($inner::perspective_infinite_reverse_rh(
                fov_y_radians,
                aspect_ratio,
                z_near,
            ))
        }

        /// Creates a right-handed orthographic projection matrix with [-1,1] depth
        /// range.  This is the same as the OpenGL `glOrtho` function in OpenGL.
        /// See
        /// https://www.khronos.org/registry/OpenGL-Refpages/gl2.1/xhtml/glOrtho.xml
        #[inline(always)]
        pub fn orthographic_rh_gl(
            left: $t,
            right: $t,
            bottom: $t,
            top: $t,
            near: $t,
            far: $t,
        ) -> Self {
            Self($inner::orthographic_rh_gl(
                left, right, bottom, top, near, far,
            ))
        }

        /// Creates a left-handed orthographic projection matrix with [0,1] depth range.
        #[inline(always)]
        pub fn orthographic_lh(
            left: $t,
            right: $t,
            bottom: $t,
            top: $t,
            near: $t,
            far: $t,
        ) -> Self {
            Self($inner::orthographic_lh(left, right, bottom, top, near, far))
        }

        /// Creates a right-handed orthographic projection matrix with [0,1] depth range.
        #[inline(always)]
        pub fn orthographic_rh(
            left: $t,
            right: $t,
            bottom: $t,
            top: $t,
            near: $t,
            far: $t,
        ) -> Self {
            Self($inner::orthographic_rh(left, right, bottom, top, near, far))
        }

        /// Transforms a 4D vector.
        #[inline(always)]
        pub fn mul_vec4(&self, other: $vec4) -> $vec4 {
            $vec4(self.0.mul_vector(&other.0))
        }

        /// Multiplies two 4x4 matrices.
        #[inline(always)]
        pub fn mul_mat4(&self, other: &Self) -> Self {
            Self(self.0.mul_matrix(&other.0))
        }

        /// Adds two 4x4 matrices.
        #[inline(always)]
        pub fn add_mat4(&self, other: &Self) -> Self {
            Self(self.0.add_matrix(&other.0))
        }

        /// Subtracts two 4x4 matrices.
        #[inline(always)]
        pub fn sub_mat4(&self, other: &Self) -> Self {
            Self(self.0.sub_matrix(&other.0))
        }

        /// Multiplies this matrix by a scalar value.
        #[inline(always)]
        pub fn mul_scalar(&self, other: $t) -> Self {
            Self(self.0.mul_scalar(other))
        }

        /// Transforms the given 3D vector as a point.
        ///
        /// This is the equivalent of multiplying the 3D vector as a 4D vector where `w` is
        /// `1.0`. Perspective correction is performed meaning the resulting `x`, `y` and `z`
        /// values are divided by `w`.
        #[inline]
        pub fn transform_point3(&self, other: $vec3) -> $vec3 {
            $vec3(self.0.transform_point3(other.0))
        }

        /// Transforms the give 3D vector as a direction.
        ///
        /// This is the equivalent of multiplying the 3D vector as a 4D vector where `w` is
        /// `0.0`.
        #[inline]
        pub fn transform_vector3(&self, other: $vec3) -> $vec3 {
            $vec3(self.0.transform_vector3(other.0))
        }

        /// Returns true if the absolute difference of all elements between `self` and `other`
        /// is less than or equal to `max_abs_diff`.
        ///
        /// This can be used to compare if two 4x4 matrices contain similar elements. It works
        /// best when comparing with a known value. The `max_abs_diff` that should be used used
        /// depends on the values being compared against.
        ///
        /// For more see
        /// [comparing floating point numbers](https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/).
        #[inline(always)]
        pub fn abs_diff_eq(&self, other: Self, max_abs_diff: $t) -> bool {
            self.0.abs_diff_eq(&other.0, max_abs_diff)
        }
    };
}

macro_rules! impl_mat4_traits {
    ($t:ty, $new:ident, $mat4:ident, $vec4:ident) => {
        /// Creates a 4x4 matrix from four column vectors.
        #[inline(always)]
        pub fn $new(x_axis: $vec4, y_axis: $vec4, z_axis: $vec4, w_axis: $vec4) -> $mat4 {
            $mat4::from_cols(x_axis, y_axis, z_axis, w_axis)
        }

        impl Default for $mat4 {
            #[inline(always)]
            fn default() -> Self {
                Self::IDENTITY
            }
        }

        impl PartialEq for $mat4 {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.as_ref().eq(other.as_ref())
            }
        }

        impl PartialOrd for $mat4 {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.as_ref().partial_cmp(other.as_ref())
            }
        }
        impl AsRef<[$t; 16]> for $mat4 {
            #[inline]
            fn as_ref(&self) -> &[$t; 16] {
                unsafe { &*(self as *const Self as *const [$t; 16]) }
            }
        }

        impl AsMut<[$t; 16]> for $mat4 {
            #[inline]
            fn as_mut(&mut self) -> &mut [$t; 16] {
                unsafe { &mut *(self as *mut Self as *mut [$t; 16]) }
            }
        }

        impl Add<$mat4> for $mat4 {
            type Output = Self;
            #[inline(always)]
            fn add(self, other: Self) -> Self {
                self.add_mat4(&other)
            }
        }

        impl Sub<$mat4> for $mat4 {
            type Output = Self;
            #[inline(always)]
            fn sub(self, other: Self) -> Self {
                self.sub_mat4(&other)
            }
        }

        impl Mul<$mat4> for $mat4 {
            type Output = Self;
            #[inline(always)]
            fn mul(self, other: Self) -> Self {
                self.mul_mat4(&other)
            }
        }

        impl Mul<$vec4> for $mat4 {
            type Output = $vec4;
            #[inline(always)]
            fn mul(self, other: $vec4) -> $vec4 {
                self.mul_vec4(other)
            }
        }

        impl Mul<$mat4> for $t {
            type Output = $mat4;
            #[inline(always)]
            fn mul(self, other: $mat4) -> $mat4 {
                other.mul_scalar(self)
            }
        }

        impl Mul<$t> for $mat4 {
            type Output = Self;
            #[inline(always)]
            fn mul(self, other: $t) -> Self {
                self.mul_scalar(other)
            }
        }

        impl Deref for $mat4 {
            type Target = Vector4x4<$vec4>;
            #[inline(always)]
            fn deref(&self) -> &Self::Target {
                unsafe { &*(self as *const Self as *const Self::Target) }
            }
        }

        impl DerefMut for $mat4 {
            #[inline(always)]
            fn deref_mut(&mut self) -> &mut Self::Target {
                unsafe { &mut *(self as *mut Self as *mut Self::Target) }
            }
        }

        #[cfg(not(target_arch = "spirv"))]
        impl fmt::Debug for $mat4 {
            fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
                fmt.debug_struct(stringify!($mat4))
                    .field("x_axis", &self.x_axis)
                    .field("y_axis", &self.y_axis)
                    .field("z_axis", &self.z_axis)
                    .field("w_axis", &self.w_axis)
                    .finish()
            }
        }

        #[cfg(not(target_arch = "spirv"))]
        impl fmt::Display for $mat4 {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(
                    f,
                    "[{}, {}, {}, {}]",
                    self.x_axis, self.y_axis, self.z_axis, self.w_axis
                )
            }
        }

        #[cfg(feature = "std")]
        impl<'a> Sum<&'a Self> for $mat4 {
            fn sum<I>(iter: I) -> Self
            where
                I: Iterator<Item = &'a Self>,
            {
                iter.fold(Self::ZERO, |a, &b| Self::add(a, b))
            }
        }

        #[cfg(feature = "std")]
        impl<'a> Product<&'a Self> for $mat4 {
            fn product<I>(iter: I) -> Self
            where
                I: Iterator<Item = &'a Self>,
            {
                iter.fold(Self::IDENTITY, |a, &b| Self::mul(a, b))
            }
        }
    };
}

#[cfg(all(target_feature = "sse2", not(feature = "scalar-math")))]
type InnerF32 = Vector4x4<__m128>;

#[cfg(any(not(target_feature = "sse2"), feature = "scalar-math"))]
type InnerF32 = Vector4x4<XYZW<f32>>;

/// A 4x4 column major matrix.
///
/// This type is 16 byte aligned if SIMD is available.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Mat4(pub(crate) InnerF32);

impl Mat4 {
    impl_mat4_methods!(f32, Vec4, Vec3, Quat, InnerF32);

    /// Transforms the given `Vec3A` as 3D point.
    ///
    /// This is the equivalent of multiplying the `Vec3A` as a 4D vector where `w` is `1.0`.
    #[inline(always)]
    pub fn transform_point3a(&self, other: Vec3A) -> Vec3A {
        Vec3A(self.0.transform_float4_as_point3(other.0))
    }

    /// Transforms the give `Vec3A` as 3D vector.
    ///
    /// This is the equivalent of multiplying the `Vec3A` as a 4D vector where `w` is `0.0`.
    #[inline(always)]
    pub fn transform_vector3a(&self, other: Vec3A) -> Vec3A {
        Vec3A(self.0.transform_float4_as_vector3(other.0))
    }

    #[inline(always)]
    pub fn as_f64(&self) -> DMat4 {
        DMat4::from_cols(
            self.x_axis.as_f64(),
            self.y_axis.as_f64(),
            self.z_axis.as_f64(),
            self.w_axis.as_f64(),
        )
    }
}
impl_mat4_traits!(f32, mat4, Mat4, Vec4);

type InnerF64 = Vector4x4<XYZW<f64>>;

/// A 4x4 column major matrix.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct DMat4(pub(crate) InnerF64);

impl DMat4 {
    impl_mat4_methods!(f64, DVec4, DVec3, DQuat, InnerF64);

    #[inline(always)]
    pub fn as_f32(&self) -> Mat4 {
        Mat4::from_cols(
            self.x_axis.as_f32(),
            self.y_axis.as_f32(),
            self.z_axis.as_f32(),
            self.w_axis.as_f32(),
        )
    }
}
impl_mat4_traits!(f64, dmat4, DMat4, DVec4);
