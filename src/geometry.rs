//! Minimal 3D vector math for DICOM patient-coordinate geometry.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[inline]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    #[inline]
    pub fn from_slice(s: &[f64]) -> Self {
        Vec3 {
            x: s[0],
            y: s[1],
            z: s[2],
        }
    }

    #[inline]
    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    #[inline]
    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3 {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }

    #[inline]
    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    #[inline]
    pub fn normalized(self) -> Vec3 {
        let l = self.length();
        if l > 1e-12 {
            Vec3 {
                x: self.x / l,
                y: self.y / l,
                z: self.z / l,
            }
        } else {
            Vec3::ZERO
        }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;
    #[inline]
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;
    #[inline]
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl std::ops::Mul<f64> for Vec3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}

/// Anatomical direction label (L/R/A/P/S/I) for the dominant axis of a
/// patient-space direction vector. Used for view edge annotations.
pub fn direction_label(v: Vec3) -> &'static str {
    let ax = v.x.abs();
    let ay = v.y.abs();
    let az = v.z.abs();
    if ax >= ay && ax >= az {
        if v.x >= 0.0 {
            "L"
        } else {
            "R"
        }
    } else if ay >= ax && ay >= az {
        if v.y >= 0.0 {
            "P"
        } else {
            "A"
        }
    } else if v.z >= 0.0 {
        "S"
    } else {
        "I"
    }
}
