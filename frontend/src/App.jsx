import { useEffect, useMemo, useRef, useState } from "react";

import * as THREE from "three";

import { Canvas, useFrame, useLoader } from "@react-three/fiber";

import { OrbitControls } from "@react-three/drei";



import {

  Search,

  MapPin,

  Clock3,

  Droplets,

  Wind,

  Gauge,

  CloudRain,

  Thermometer,

  Eye,

  AlertTriangle,

  ChevronRight,

  RefreshCw,

  Brain,

  Activity,

} from "lucide-react";



import {

  ResponsiveContainer,

  LineChart,

  Line,

  XAxis,

  YAxis,

  Tooltip,

  CartesianGrid,

} from "recharts";



import "./App.css";



const API = "http://127.0.0.1:8000";



/* =========================================================

   CITY DATA

   ========================================================= */



const CITY_COORDS = {

  Delhi: [28.6139, 77.209],

  Mumbai: [19.076, 72.8777],

  Bengaluru: [12.9716, 77.5946],

  Chennai: [13.0827, 80.2707],

  Hyderabad: [17.385, 78.4867],

  Kolkata: [22.5726, 88.3639],

  Pune: [18.5204, 73.8567],

  Ahmedabad: [23.0225, 72.5714],

  Jaipur: [26.9124, 75.7873],

  Lucknow: [26.8467, 80.9462],

  Patna: [25.5941, 85.1376],

  Bhopal: [23.2599, 77.4126],

  Chandigarh: [30.7333, 76.7794],

  Guwahati: [26.1445, 91.7362],

  Bhubaneswar: [20.2961, 85.8245],

  Kochi: [9.9312, 76.2673],

  Indore: [22.7196, 75.8577],

  Nagpur: [21.1458, 79.0882],

  Surat: [21.1702, 72.8311],

  Visakhapatnam: [17.6868, 83.2185],

};



/* =========================================================

   GEO

   ========================================================= */



function latLongToVector3(lat, lon, radius = 2.52) {

  const phi = (90 - lat) * (Math.PI / 180);

  const theta = (lon + 180) * (Math.PI / 180);



  return new THREE.Vector3(

    -radius * Math.sin(phi) * Math.cos(theta),

    radius * Math.cos(phi),

    radius * Math.sin(phi) * Math.sin(theta)

  );

}



/* =========================================================

   WEATHER HELPERS

   ========================================================= */



function weatherIcon(code, isDay = true) {

  const value = Number(code);



  if (value === 0) return isDay ? "☀️" : "🌙";

  if (value === 1 || value === 2) return isDay ? "🌤️" : "🌙";

  if (value === 3) return "☁️";

  if ([45, 48].includes(value)) return "🌫️";

  if ([51, 53, 55].includes(value)) return "🌦️";

  if ([56, 57].includes(value)) return "🌧️";

  if ([61, 63, 65].includes(value)) return "🌧️";

  if ([66, 67].includes(value)) return "🌧️";

  if ([71, 73, 75, 77].includes(value)) return "🌨️";

  if ([80, 81, 82].includes(value)) return "🌦️";

  if ([85, 86].includes(value)) return "🌨️";

  if ([95, 96, 99].includes(value)) return "⛈️";



  return "🌤️";

}



function weatherDescription(code) {

  const value = Number(code);



  if (value === 0) return "Clear sky";

  if (value === 1) return "Mainly clear";

  if (value === 2) return "Partly cloudy";

  if (value === 3) return "Overcast";

  if ([45, 48].includes(value)) return "Fog";

  if ([51, 53, 55].includes(value)) return "Drizzle";

  if ([56, 57].includes(value)) return "Freezing drizzle";

  if ([61, 63, 65].includes(value)) return "Rain";

  if ([66, 67].includes(value)) return "Freezing rain";

  if ([71, 73, 75, 77].includes(value)) return "Snow";

  if ([80, 81, 82].includes(value)) return "Rain showers";

  if ([85, 86].includes(value)) return "Snow showers";

  if ([95, 96, 99].includes(value)) return "Thunderstorm";



  return "Weather conditions";

}



function formatHour(time) {

  if (!time) return "--";



  const hour = Number(time.slice(11, 13));

  const suffix = hour >= 12 ? "PM" : "AM";

  const h = hour % 12 || 12;



  return `${h} ${suffix}`;

}



function getAqiLabel(value) {

  const aqi = Number(value);



  if (!Number.isFinite(aqi)) return "Unavailable";

  if (aqi <= 50) return "Good";

  if (aqi <= 100) return "Moderate";

  if (aqi <= 150) return "Unhealthy for sensitive groups";

  if (aqi <= 200) return "Unhealthy";

  if (aqi <= 300) return "Very unhealthy";



  return "Hazardous";

}



/* =========================================================

   SAFE NUMBER HELPERS

   ========================================================= */



function toNumber(value) {

  if (value === null || value === undefined || value === "") {

    return null;

  }



  const number = Number(value);



  return Number.isFinite(number) ? number : null;

}



function findValue(object, possibleKeys, visited = new Set()) {

  if (

    object === null ||

    object === undefined ||

    typeof object !== "object"

  ) {

    return null;

  }



  if (visited.has(object)) return null;



  visited.add(object);



  for (const key of possibleKeys) {

    if (Object.prototype.hasOwnProperty.call(object, key)) {

      const value = object[key];



      if (value !== null && value !== undefined && value !== "") {

        return value;

      }

    }

  }



  for (const value of Object.values(object)) {

    if (value && typeof value === "object") {

      const result = findValue(value, possibleKeys, visited);



      if (result !== null && result !== undefined) {

        return result;

      }

    }

  }



  return null;

}



/* =========================================================

   ML RESPONSE NORMALIZATION

   ========================================================= */



function normalizePrediction(data) {

  const temperature = toNumber(

    findValue(data, [

      "predicted_temperature",

      "next_hour_temperature",

      "prediction_temperature",

      "temperature_prediction",

      "predicted_temp",

      "next_temperature",

      "temperature",

    ])

  );



  const rainProbability = toNumber(

    findValue(data, [

      "rain_probability",

      "next_hour_rain_probability",

      "predicted_rain_probability",

      "rain_probability_percent",

      "rain_probability_pct",

      "precipitation_probability",

    ])

  );



  let rainExpected = findValue(data, [

    "rain_expected",

    "next_hour_rain",

    "predicted_rain",

    "rain_prediction",

    "will_rain",

  ]);



  if (

    rainExpected === null &&

    rainProbability !== null

  ) {

    rainExpected = rainProbability >= 50;

  }



  if (typeof rainExpected === "string") {

    const normalized = rainExpected.toLowerCase();



    if (normalized === "yes" || normalized === "true") {

      rainExpected = true;

    } else if (

      normalized === "no" ||

      normalized === "false"

    ) {

      rainExpected = false;

    }

  }



  return {

    raw: data,

    temperature,

    rainProbability,

    rainExpected:

      rainExpected === true

        ? true

        : rainExpected === false

        ? false

        : null,

  };

}



/* =========================================================

   RISK

   ========================================================= */



function getRiskLevel(weather, hourly = []) {

  const rain = Number(weather?.rain_probability ?? 0);

  const wind = Number(weather?.wind_speed ?? 0);

  const code = Number(weather?.weather_code ?? 0);



  const maxRain = hourly.length

    ? Math.max(

        ...hourly.map((item) =>

          Number(item.rainProbability || 0)

        )

      )

    : rain;



  if (

    [95, 96, 99].includes(code) ||

    wind >= 60 ||

    maxRain >= 85

  ) {

    return {

      label: "ELEVATED",

      className: "elevated",

      description: "Severe-weather indicators detected",

    };

  }



  if (

    rain >= 60 ||

    maxRain >= 70 ||

    wind >= 40

  ) {

    return {

      label: "WATCH",

      className: "watch",

      description: "Monitor changing conditions",

    };

  }



  return {

    label: "NORMAL",

    className: "normal",

    description: "No major indicators detected",

  };

}



/* =========================================================

   EARTH

   ========================================================= */



function Earth() {

  const texture = useLoader(

    THREE.TextureLoader,

    "https://threejs.org/examples/textures/planets/earth_atmos_2048.jpg"

  );



  return (

    <group>

      <mesh>

        <sphereGeometry args={[2.5, 96, 96]} />



        <meshStandardMaterial

          map={texture}

          roughness={0.9}

          metalness={0.02}

        />

      </mesh>



      <mesh>

        <sphereGeometry args={[2.56, 96, 96]} />



        <meshBasicMaterial

          color="#38bdf8"

          transparent

          opacity={0.075}

          side={THREE.BackSide}

        />

      </mesh>

    </group>

  );

}



/* =========================================================

   ONE SELECTED PIN ONLY

   ========================================================= */



function CityMarker({ city }) {
  const coords = CITY_COORDS[city];

  if (!coords) return null;

  const position = latLongToVector3(
    coords[0],
    coords[1],
    2.56
  );

  return (
    <mesh position={position}>
      <sphereGeometry args={[0.055, 20, 20]} />
      <meshBasicMaterial color="#38bdf8" />
    </mesh>
  );
}



/* =========================================================

   GLOBE SCENE

   ========================================================= */



function GlobeScene({

  selectedCity,

  resetSignal,

}) {

  const controlsRef = useRef(null);



  const targetCamera = useRef(

    new THREE.Vector3(0, 0, 7.2)

  );



  const flying = useRef(false);



  const focusCity = (city) => {

    const coords = CITY_COORDS[city];



    if (!coords) return;



    const direction = latLongToVector3(

      coords[0],

      coords[1],

      1

    ).normalize();



    targetCamera.current

      .copy(direction)

      .multiplyScalar(3.65);



    flying.current = true;



    if (controlsRef.current) {

      controlsRef.current.enabled = false;

    }

  };



  useEffect(() => {

    if (!selectedCity) return;



    focusCity(selectedCity);

  }, [selectedCity]);



  useEffect(() => {

    if (resetSignal === 0) return;



    focusCity("Bengaluru");

  }, [resetSignal]);



  useFrame(({ camera }) => {

    if (!flying.current) return;



    camera.position.lerp(

      targetCamera.current,

      0.065

    );



    camera.lookAt(0, 0, 0);



    const distance =

      camera.position.distanceTo(

        targetCamera.current

      );



    if (distance < 0.015) {

      camera.position.copy(

        targetCamera.current

      );



      camera.lookAt(0, 0, 0);



      flying.current = false;



      if (controlsRef.current) {

        controlsRef.current.target.set(

          0,

          0,

          0

        );



        controlsRef.current.enabled = true;

        controlsRef.current.update();

      }

    }

  });



  return (

    <>

      <ambientLight intensity={1.15} />



      <directionalLight

        position={[5, 5, 5]}

        intensity={2}

      />



      <pointLight

        position={[-5, -2, 4]}

        intensity={0.5}

      />



      <Earth />



      {/* EXACTLY ONE PIN */}

      {selectedCity && (

        <CityMarker city={selectedCity} />

      )}



      <OrbitControls

        ref={controlsRef}

        enablePan={false}

        enableDamping

        dampingFactor={0.07}

        rotateSpeed={0.4}

        zoomSpeed={0.65}

        minDistance={3.15}

        maxDistance={8.5}

        target={[0, 0, 0]}

      />

    </>

  );

}



/* =========================================================

   GLOBE

   ========================================================= */



function Globe({

  selectedCity,

  resetSignal,

}) {

  return (

    <Canvas

      camera={{

        position: [0, 0, 7.2],

        fov: 42,

        near: 0.1,

        far: 100,

      }}

      dpr={[1, 2]}

      gl={{

        antialias: true,

        alpha: true,

      }}

    >

      <color

        attach="background"

        args={["#030712"]}

      />



      <GlobeScene

        selectedCity={selectedCity}

        resetSignal={resetSignal}

      />

    </Canvas>

  );

}



/* =========================================================

   METRIC

   ========================================================= */



function Metric({

  icon: Icon,

  label,

  value,

  unit,

}) {

  return (

    <div className="metric">

      <div className="metric-icon">

        <Icon size={16} />

      </div>



      <div>

        <span>{label}</span>



        <strong>

          {value}

          {unit && <small>{unit}</small>}

        </strong>

      </div>

    </div>

  );

}



/* =========================================================

   HOURLY CARD

   ========================================================= */



function HourlyCard({ item, active }) {

  return (

    <div

      className={`hour-card ${

        active ? "active" : ""

      }`}

    >

      <span className="hour-time">

        {item.label}

      </span>



      <div className="hour-icon">

        {weatherIcon(

          item.weatherCode,

          item.isDay

        )}

      </div>



      <strong className="hour-temp">

        {Math.round(item.temperature)}°

      </strong>



      <div className="hour-rain">

        <CloudRain size={13} />

        {item.rainProbability}%

      </div>



      <span className="hour-mm">

        {Number(item.precipitation).toFixed(1)} mm

      </span>

    </div>

  );

}



/* =========================================================

   24-HOUR CHART

   ========================================================= */



function HourlyChart({ hourly }) {

  if (!hourly.length) return null;



  const chartData = hourly.map((item) => ({

    time: item.label,

    temperature: Number(

      Number(item.temperature).toFixed(1)

    ),

    rain: Number(item.rainProbability),

  }));



  return (

    <div

      className="hourly-chart"

      style={{

        width: "100%",

        minWidth: 0,

        overflow: "hidden",

        boxSizing: "border-box",

      }}

    >

      <div className="chart-title-row">

        <div>

          <span className="section-kicker">

            24-HOUR OUTLOOK

          </span>



          <h3>

            Temperature & rain probability

          </h3>

        </div>



        <div className="chart-legend">

          <span>

            <i className="legend-temp" />

            Temperature

          </span>



          <span>

            <i className="legend-rain" />

            Rain %

          </span>

        </div>

      </div>



      <div className="chart-container">

        <ResponsiveContainer

          width="100%"

          height="100%"

          minWidth={1}

          minHeight={280}

        >

          <LineChart

            data={chartData}

            margin={{

              top: 15,

              right: 15,

              left: 5,

              bottom: 5,

            }}

          >

            <CartesianGrid

              strokeDasharray="3 3"

              stroke="rgba(255,255,255,0.08)"

            />



            <XAxis

              dataKey="time"

              tick={{

                fill: "#64748b",

                fontSize: 10,

              }}

              interval={2}

              axisLine={false}

              tickLine={false}

            />



            <YAxis

              yAxisId="temp"

              orientation="left"

              tick={{

                fill: "#64748b",

                fontSize: 10,

              }}

              tickFormatter={(value) => `${value}°`}

              axisLine={false}

              tickLine={false}

              width={42}

            />



            <YAxis

              yAxisId="rain"

              orientation="right"

              domain={[0, 100]}

              tick={{

                fill: "#64748b",

                fontSize: 10,

              }}

              tickFormatter={(value) => `${value}%`}

              axisLine={false}

              tickLine={false}

              width={42}

            />



            <Tooltip

              wrapperStyle={{

                zIndex: 1000,

                outline: "none",

              }}

              contentStyle={{

                background: "#07111f",

                border:

                  "1px solid rgba(255,255,255,0.12)",

                borderRadius: 10,

                color: "#fff",

                fontSize: 11,

              }}

              labelStyle={{

                color: "#cbd5e1",

                marginBottom: 5,

              }}

              formatter={(value, name) => {

                if (name === "Temperature") {

                  return [

                    `${Number(value).toFixed(1)}°C`,

                    name,

                  ];

                }



                return [

                  `${Number(value).toFixed(0)}%`,

                  name,

                ];

              }}

            />



            <Line

              yAxisId="temp"

              type="monotone"

              dataKey="temperature"

              stroke="#38bdf8"

              strokeWidth={3}

              dot={{ r: 2 }}

              activeDot={{ r: 5 }}

              connectNulls

              name="Temperature"

            />



            <Line

              yAxisId="rain"

              type="monotone"

              dataKey="rain"

              stroke="#a78bfa"

              strokeWidth={2}

              strokeDasharray="5 5"

              dot={{ r: 2 }}

              activeDot={{ r: 4 }}

              connectNulls

              name="Rain probability"

            />

          </LineChart>

        </ResponsiveContainer>

      </div>

    </div>

  );

}



function HazardAlertDock({

  risk,

  maxRain,

  maxRainHour,

  onOpen,

}) {

  if (

    !risk ||

    risk.className === "normal"

  ) {

    return null;

  }



  const critical =

    risk.className === "elevated";



  return (

    <button

      className={`hazard-alert-dock ${

        critical ? "critical" : "watch"

      }`}

      onClick={onOpen}

    >

      <span className="hazard-alert-pulse" />



      <span className="hazard-alert-icon">

        <AlertTriangle size={17} />

      </span>



      <span className="hazard-alert-copy">

        <strong>

          {critical

            ? "WEATHER ALERT"

            : "WEATHER WATCH"}

        </strong>



        <small>

          {critical

            ? "Elevated atmospheric indicators"

            : "Changing conditions detected"}

        </small>

      </span>



      <span className="hazard-alert-value">

        <b>{maxRain}%</b>



        <small>

          {maxRainHour?.label

            ? `rain • ${maxRainHour.label}`

            : "rain risk"}

        </small>

      </span>



      <ChevronRight size={15} />

    </button>

  );

}



/* =========================================================

   APP

   ========================================================= */



export default function App() {

  const [cities, setCities] = useState({});



  /* DEFAULT CITY */

  const [selectedCity, setSelectedCity] =

    useState("Bengaluru");



  const [search, setSearch] = useState("");



  const [prediction, setPrediction] =

    useState(null);



  const [forecast, setForecast] = useState([]);



  const [hourly, setHourly] = useState([]);



  const [

    timezoneAbbreviation,

    setTimezoneAbbreviation,

  ] = useState("");



  const [loadingHourly, setLoadingHourly] =

    useState(false);



  const [

    loadingPrediction,

    setLoadingPrediction,

  ] = useState(false);



  const [airQuality, setAirQuality] =

    useState(null);



  const [resetSignal, setResetSignal] =

    useState(0);



  const hazardRef = useRef(null);



  /* =====================================================

     SEARCH

     ===================================================== */



  const filteredCities = useMemo(() => {

    if (!search.trim()) return [];



    return Object.keys(CITY_COORDS)

      .filter((city) =>

        city

          .toLowerCase()

          .includes(search.toLowerCase())

      )

      .slice(0, 7);

  }, [search]);



  const selectedWeather =

    cities[selectedCity];



  /* =====================================================

     LIVE WEATHER

     ===================================================== */



  const loadCities = async () => {

    try {

      const response = await fetch(

        `${API}/cities/weather`

      );



      if (!response.ok) {

        throw new Error(

          "Failed to load city weather"

        );

      }



      const data = await response.json();



      const map = {};



      if (Array.isArray(data)) {

        data.forEach((item) => {

          map[item.city] = item;

        });

      } else if (data?.cities) {

        data.cities.forEach((item) => {

          map[item.city] = item;

        });

      }



      setCities(map);

    } catch (error) {

      console.error(

        "City weather error:",

        error

      );

    }

  };



  /* =====================================================

     ML

     ===================================================== */



  const loadPrediction = async (city) => {

    try {

      setLoadingPrediction(true);



      const response = await fetch(

        `${API}/predict/${encodeURIComponent(city)}`

      );



      if (!response.ok) {

        throw new Error(

          "Prediction failed"

        );

      }



      const data = await response.json();



      /*

       * IMPORTANT:

       * Normalize backend response so the UI does not

       * accidentally display 0.0 when the backend uses

       * a different field name.

       */

      setPrediction(

        normalizePrediction(data)

      );

    } catch (error) {

      console.error(

        "Prediction error:",

        error

      );



      setPrediction(null);

    } finally {

      setLoadingPrediction(false);

    }

  };



  /* =====================================================

     FORECAST

     ===================================================== */



  const loadForecast = async (city) => {

    try {

      const response = await fetch(

        `${API}/forecast/${encodeURIComponent(city)}`

      );



      if (!response.ok) {

        throw new Error(

          "Forecast failed"

        );

      }



      const data = await response.json();



      setForecast(

        data.forecast || []

      );

    } catch (error) {

      console.error(

        "Forecast error:",

        error

      );



      setForecast([]);

    }

  };



  /* =====================================================

     HOURLY

     ===================================================== */



  const loadHourly = async (city) => {

    const coords = CITY_COORDS[city];



    if (!coords) return;



    try {

      setLoadingHourly(true);



      const [

        latitude,

        longitude,

      ] = coords;



      const params = new URLSearchParams({

        latitude,

        longitude,



        hourly: [

          "temperature_2m",

          "relative_humidity_2m",

          "apparent_temperature",

          "precipitation_probability",

          "precipitation",

          "rain",

          "weather_code",

          "wind_speed_10m",

          "wind_gusts_10m",

          "cloud_cover",

          "is_day",

        ].join(","),



        forecast_hours: "24",

        timezone: "auto",

        temperature_unit: "celsius",

        wind_speed_unit: "kmh",

        precipitation_unit: "mm",

      });



      const response = await fetch(

        `https://api.open-meteo.com/v1/forecast?${params.toString()}`

      );



      if (!response.ok) {

        throw new Error(

          "Hourly weather failed"

        );

      }



      const data = await response.json();



      setTimezoneAbbreviation(

        data.timezone_abbreviation || ""

      );



      const times =

        data.hourly?.time || [];



      const result = times.map(

        (time, index) => ({

          time,



          label: formatHour(time),



          temperature:

            data.hourly

              ?.temperature_2m?.[index] ?? 0,



          humidity:

            data.hourly

              ?.relative_humidity_2m?.[index] ?? 0,



          apparent:

            data.hourly

              ?.apparent_temperature?.[index] ?? 0,



          rainProbability:

            data.hourly

              ?.precipitation_probability?.[

                index

              ] ?? 0,



          precipitation:

            data.hourly

              ?.precipitation?.[index] ?? 0,



          rain:

            data.hourly

              ?.rain?.[index] ?? 0,



          weatherCode:

            data.hourly

              ?.weather_code?.[index] ?? 0,



          wind:

            data.hourly

              ?.wind_speed_10m?.[index] ?? 0,



          gusts:

            data.hourly

              ?.wind_gusts_10m?.[index] ?? 0,



          cloud:

            data.hourly

              ?.cloud_cover?.[index] ?? 0,



          isDay: Boolean(

            data.hourly

              ?.is_day?.[index]

          ),

        })

      );



      setHourly(result);

    } catch (error) {

      console.error(

        "Hourly error:",

        error

      );



      setHourly([]);

    } finally {

      setLoadingHourly(false);

    }

  };



  /* =====================================================

     AQI

     ===================================================== */



  const loadAirQuality = async (city) => {

    const coords = CITY_COORDS[city];



    if (!coords) return;



    try {

      const [

        latitude,

        longitude,

      ] = coords;



      const params = new URLSearchParams({

        latitude,

        longitude,



        current:

          "us_aqi,european_aqi,pm2_5,pm10,ozone,nitrogen_dioxide",



        timezone: "auto",

      });



      const response = await fetch(

        `https://air-quality-api.open-meteo.com/v1/air-quality?${params.toString()}`

      );



      if (!response.ok) {

        throw new Error(

          "AQI failed"

        );

      }



      const data = await response.json();



      setAirQuality(

        data.current || null

      );

    } catch (error) {

      console.error(

        "AQI error:",

        error

      );



      setAirQuality(null);

    }

  };



  /* =====================================================

     INITIAL

     ===================================================== */



  useEffect(() => {

    loadCities();



    const interval = setInterval(

      loadCities,

      5 * 60 * 1000

    );



    return () =>

      clearInterval(interval);

  }, []);



  /* =====================================================

     CITY CHANGE

     ===================================================== */



  useEffect(() => {

    if (!selectedCity) return;



    loadPrediction(selectedCity);

    loadForecast(selectedCity);

    loadHourly(selectedCity);

    loadAirQuality(selectedCity);

  }, [selectedCity]);



  /* =====================================================

     SELECT CITY

     ===================================================== */



  const selectCity = (city) => {

    setSelectedCity(city);

    setSearch("");

  };



  /* =====================================================

     RESET GLOBE

     ===================================================== */



  const resetGlobe = () => {

    setSelectedCity("Bengaluru");

    setSearch("");



    setResetSignal(

      (value) => value + 1

    );

  };



  /* =====================================================

     ANALYTICS

     ===================================================== */



  const risk = getRiskLevel(

    selectedWeather,

    hourly

  );



  const maxRain = hourly.length

    ? Math.max(

        ...hourly.map(

          (item) =>

            Number(

              item.rainProbability || 0

            )

        )

      )

    : 0;



  const maxRainHour = hourly.find(

    (item) =>

      Number(

        item.rainProbability || 0

      ) === maxRain

  );



  const totalRain = hourly.reduce(

    (sum, item) =>

      sum +

      Number(

        item.precipitation || 0

      ),

    0

  );



  const warmest = hourly.length

    ? hourly.reduce((a, b) =>

        a.temperature > b.temperature

          ? a

          : b

      )

    : null;



  const coolest = hourly.length

    ? hourly.reduce((a, b) =>

        a.temperature < b.temperature

          ? a

          : b

      )

    : null;



  /* =====================================================

     RENDER

     ===================================================== */



  return (

    <div className="app">



      {/* =================================================

          HEADER

          ================================================= */}



      <header className="topbar">



        <div className="brand">

          <div className="brand-mark">

            <Activity size={21} />

          </div>



          <div>

            <strong>

              WeatherSense

            </strong>



            <span>

              AI WEATHER INTELLIGENCE

            </span>

          </div>

        </div>



        <div className="search-wrapper">



          <Search size={18} />



          <input

            value={search}

            onChange={(e) =>

              setSearch(e.target.value)

            }

            onKeyDown={(e) => {

              if (

                e.key === "Enter" &&

                filteredCities.length

              ) {

                selectCity(

                  filteredCities[0]

                );

              }

            }}

            placeholder="Search city..."

          />



          {search &&

            filteredCities.length > 0 && (

              <div className="search-results">



                {filteredCities.map(

                  (city) => (

                    <button

                      key={city}

                      onClick={() =>

                        selectCity(city)

                      }

                    >

                      <MapPin size={15} />



                      <span>

                        {city}

                      </span>



                      <ChevronRight

                        size={14}

                      />

                    </button>

                  )

                )}



              </div>

            )}

        </div>



        <div className="live-status">

          <span className="live-dot" />

          LIVE

        </div>

      </header>



      {/* =================================================

          MAIN

          ================================================= */}



      <main className="dashboard">



        {/* =================================================

            GLOBE

            ================================================= */}



        <section className="globe-section">



          <div className="globe-header">



            <span className="eyebrow">

              LIVE WEATHER MAP

            </span>



            <h1>

              {selectedCity}

            </h1>



            <p>

              Selected location • drag to explore

            </p>



          </div>



          <div

            className="globe-container"

            style={{

              position: "relative",

              overflow: "hidden",

            }}

          >



            <Globe

              selectedCity={selectedCity}

              resetSignal={resetSignal}

            />



            {/* FIXED CITY INFO — NOT 3D */}



            <div className="selected-location">



              <MapPin size={14} />



              <strong>

                {selectedCity}

              </strong>



              {selectedWeather && (

                <span>

                  {Math.round(

                    selectedWeather.temperature ??

                      0

                  )}

                  °

                </span>

              )}



            </div>



            <div className="globe-overlay">



              <button

                className="reset-view"

                onClick={resetGlobe}

              >

                <RefreshCw size={14} />

                Bengaluru

              </button>



            </div>



            



          </div>

        </section>



        {/* =================================================

            SIDE PANEL

            ================================================= */}



        <aside className="side-panel">



          {selectedWeather ? (

            <>



              {/* CURRENT CONDITIONS */}



              <div

                className="panel-city"

                style={{

                  minWidth: 0,

                  overflow: "hidden",

                }}

              >



                <div

                  className="panel-city-top"

                  style={{

                    display: "flex",

                    alignItems: "center",

                    justifyContent:

                      "space-between",

                    gap: 12,

                    minWidth: 0,

                  }}

                >



                  <div

                    style={{

                      minWidth: 0,

                      flex: 1,

                    }}

                  >



                    <span className="panel-kicker">

                      CURRENT CONDITIONS

                    </span>



                    <h2

                      style={{

                        overflow: "hidden",

                        textOverflow:

                          "ellipsis",

                        whiteSpace:

                          "nowrap",

                      }}

                    >

                      {selectedCity}

                    </h2>



                    <div className="coordinates">

                      <MapPin size={13} />



                      {CITY_COORDS[

                        selectedCity

                      ][0].toFixed(4)}



                      °,{" "}



                      {CITY_COORDS[

                        selectedCity

                      ][1].toFixed(4)}



                      °

                    </div>



                  </div>



                  <div

                    className="weather-big-icon"

                    style={{

                      flexShrink: 0,

                      width: 48,

                      height: 48,

                      display: "flex",

                      alignItems: "center",

                      justifyContent:

                        "center",

                      fontSize: 30,

                    }}

                  >

                    {weatherIcon(

                      selectedWeather.weather_code,

                      true

                    )}

                  </div>



                </div>



                <div

                  className="temperature-row"

                  style={{

                    display: "flex",

                    alignItems: "center",

                    gap: 12,

                    minWidth: 0,

                    marginTop: 10,

                  }}

                >



                  <strong

                    style={{

                      flexShrink: 0,

                      whiteSpace:

                        "nowrap",

                    }}

                  >

                    {Math.round(

                      selectedWeather.temperature ??

                        0

                    )}

                    °

                  </strong>



                  <div

                    style={{

                      minWidth: 0,

                    }}

                  >



                    <span

                      style={{

                        display: "block",

                        overflow:

                          "hidden",

                        textOverflow:

                          "ellipsis",

                        whiteSpace:

                          "nowrap",

                      }}

                    >

                      {weatherDescription(

                        selectedWeather.weather_code

                      )}

                    </span>



                    <small

                      style={{

                        display: "block",

                        overflow:

                          "hidden",

                        textOverflow:

                          "ellipsis",

                        whiteSpace:

                          "nowrap",

                      }}

                    >

                      Feels like{" "}

                      {Math.round(

                        selectedWeather.apparent_temperature ??

                          selectedWeather.temperature ??

                          0

                      )}

                      °C

                    </small>



                  </div>



                </div>



                <div

                  className="local-time"

                  style={{

                    display: "flex",

                    alignItems: "center",

                    gap: 7,

                    minWidth: 0,

                  }}

                >

                  <Clock3 size={15} />



                  <span

                    style={{

                      overflow:

                        "hidden",

                      textOverflow:

                        "ellipsis",

                      whiteSpace:

                        "nowrap",

                    }}

                  >

                    Local time

                  </span>



                  <strong

                    style={{

                      marginLeft:

                        "auto",

                      flexShrink: 0,

                    }}

                  >

                    {timezoneAbbreviation ||

                      "--"}

                  </strong>

                </div>



                <div className="metrics-grid">



                  <Metric

                    icon={Droplets}

                    label="Humidity"

                    value={Math.round(

                      selectedWeather.humidity ??

                        0

                    )}

                    unit="%"

                  />



                  <Metric

                    icon={Wind}

                    label="Wind"

                    value={Math.round(

                      selectedWeather.wind_speed ??

                        0

                    )}

                    unit=" km/h"

                  />



                  <Metric

                    icon={Gauge}

                    label="Pressure"

                    value={Math.round(

                      selectedWeather.pressure ??

                        0

                    )}

                    unit=" hPa"

                  />



                  <Metric

                    icon={CloudRain}

                    label="Rain"

                    value={Number(

                      selectedWeather.rain ??

                        0

                    ).toFixed(1)}

                    unit=" mm"

                  />



                </div>



              </div>



              {/* =================================================

                  ML

                  ================================================= */}



              <div

                className="ai-card"

                style={{

                  minWidth: 0,

                  overflow: "hidden",

                }}

              >



                <div className="ai-header">



                  <div className="ai-title">



                    <div className="ai-icon">

                      <Brain size={18} />

                    </div>



                    <div>

                      <span>

                        WEATHERSENSE ML

                      </span>



                      <strong>

                        NEXT-HOUR PREDICTION

                      </strong>

                    </div>



                  </div>



                  <span className="ai-badge">

                    AI

                  </span>



                </div>



                {loadingPrediction ? (

                  <div className="loading-state">

                    Running WeatherSense ML...

                  </div>

                ) : prediction ? (

                  <>



                    <div

                      className="prediction-next-time"

                      style={{

                        marginTop: 9,

                      }}

                    >

                      Next hour

                      {hourly[0]?.label

                        ? ` • ${hourly[0].label}`

                        : ""}

                    </div>



                    <div

                      className="prediction-grid"

                      style={{

                        width: "100%",

                        minWidth: 0,

                        display: "grid",

                        gridTemplateColumns:

                          "repeat(3, minmax(0, 1fr))",

                        gap: 7,

                      }}

                    >



                      <div

                        style={{

                          minWidth: 0,

                          overflow: "hidden",

                        }}

                      >

                        <span

                          style={{

                            display: "block",

                            overflow: "hidden",

                            textOverflow:

                              "ellipsis",

                            whiteSpace:

                              "nowrap",

                          }}

                        >

                          Temperature

                        </span>



                        <strong

                          style={{

                            display: "block",

                            overflow: "hidden",

                            textOverflow:

                              "ellipsis",

                            whiteSpace:

                              "nowrap",

                          }}

                        >

                          {prediction.temperature !=

                          null

                            ? Number(

                                prediction.temperature

                              ).toFixed(1)

                            : "--"}

                          °C

                        </strong>

                      </div>



                      <div

                        style={{

                          minWidth: 0,

                          overflow: "hidden",

                        }}

                      >

                        <span

                          style={{

                            display: "block",

                            overflow: "hidden",

                            textOverflow:

                              "ellipsis",

                            whiteSpace:

                              "nowrap",

                          }}

                        >

                          Rain probability

                        </span>



                        <strong

                          style={{

                            display: "block",

                            overflow: "hidden",

                            textOverflow:

                              "ellipsis",

                            whiteSpace:

                              "nowrap",

                          }}

                        >

                          {prediction.rainProbability !=

                          null

                            ? Number(

                                prediction.rainProbability

                              ).toFixed(1)

                            : "--"}

                          %

                        </strong>

                      </div>



                      <div

                        style={{

                          minWidth: 0,

                          overflow: "hidden",

                        }}

                      >

                        <span

                          style={{

                            display: "block",

                            overflow: "hidden",

                            textOverflow:

                              "ellipsis",

                            whiteSpace:

                              "nowrap",

                          }}

                        >

                          Rain expected

                        </span>



                        <strong

                          className={

                            prediction.rainExpected ===

                            true

                              ? "rain-yes"

                              : prediction.rainExpected ===

                                false

                              ? "rain-no"

                              : ""

                          }

                          style={{

                            display: "block",

                            overflow: "hidden",

                            textOverflow:

                              "ellipsis",

                            whiteSpace:

                              "nowrap",

                          }}

                        >

                          {prediction.rainExpected ==

                          null

                            ? "--"

                            : prediction.rainExpected

                            ? "YES"

                            : "NO"}

                        </strong>

                      </div>



                    </div>



                    <div className="ai-explanation">

                      <Brain size={12} />



                      <span>

                        WeatherSense ML predicts

                        next-hour temperature and

                        rain probability using

                        current and historical

                        weather features.

                      </span>

                    </div>



                  </>

                ) : (

                  <div className="loading-state">

                    Prediction unavailable

                  </div>

                )}



              </div>



              {/* =================================================

                  ALERT

                  ================================================= */}



              <div

                className={`compact-alert-card ${risk.className}`}

                ref={hazardRef}

              >



                <div className="compact-alert-icon">

                  <AlertTriangle size={18} />

                </div>



                <div className="compact-alert-copy">



                  <span>

                    WEATHER ALERT

                  </span>



                  <strong>

                    {risk.label}

                  </strong>



                  <small>

                    {risk.description}

                  </small>



                </div>



                <div className="compact-alert-stat">



                  <b>

                    {maxRain}%

                  </b>



                  <small>

                    peak rain

                  </small>



                </div>



              </div>



              {/* =================================================

                  AQI

                  ================================================= */}



              <div className="aqi-card">



                <div className="aqi-top">



                  <div className="aqi-icon">

                    <Activity size={18} />

                  </div>



                  <div>

                    <span>

                      AIR QUALITY

                    </span>



                    <strong>

                      US AQI

                    </strong>

                  </div>



                  <b>

                    {airQuality?.us_aqi ??

                      "--"}

                  </b>



                </div>



                <div className="aqi-status">



                  <span>

                    {getAqiLabel(

                      airQuality?.us_aqi

                    )}

                  </span>



                  <small>

                    PM2.5{" "}

                    {airQuality?.pm2_5 !=

                    null

                      ? Number(

                          airQuality.pm2_5

                        ).toFixed(1)

                      : "--"}{" "}

                    µg/m³

                  </small>



                </div>



              </div>



              <div className="side-source">

                <Eye size={13} />



                Live weather + air quality:

                Open-Meteo



                <span>•</span>



                ML: WeatherSense

              </div>



            </>

          ) : (

            <div className="empty-panel">

              <MapPin size={28} />



              <h2>

                Loading weather...

              </h2>

            </div>

          )}



        </aside>

      </main>



      {/* =================================================

          FORECAST

          ================================================= */}



      {selectedWeather && (

        <section className="forecast-lower">



          <div className="forecast-lower-header">



            <div>



              <span className="eyebrow">

                WHAT HAPPENS NEXT

              </span>



              <h2>

                {selectedCity} forecast

                intelligence

              </h2>



              <p>

                Detailed 24-hour conditions

                and extended outlook.

              </p>



            </div>



            {loadingHourly && (

              <span className="loading-mini">

                Updating...

              </span>

            )}



          </div>



          {/* =================================================

              24 HOURS

              ================================================= */}



          <div className="lower-section">



            <div className="section-header">



              <div>



                <span className="section-kicker">

                  SHORT RANGE

                </span>



                <h3>

                  Next 24 hours

                </h3>



              </div>



            </div>



            <div className="hourly-summary">



              <div>

                <CloudRain size={15} />



                <span>

                  Peak rain

                </span>



                <strong>

                  {maxRain}%

                </strong>



                <small>

                  {maxRainHour?.label ||

                    "--"}

                </small>

              </div>



              <div>

                <Droplets size={15} />



                <span>

                  Total rain

                </span>



                <strong>

                  {totalRain.toFixed(1)} mm

                </strong>



                <small>

                  next 24h

                </small>

              </div>



              <div>

                <Thermometer size={15} />



                <span>

                  Range

                </span>



                <strong>

                  {coolest

                    ? `${Math.round(

                        coolest.temperature

                      )}° → ${Math.round(

                        warmest.temperature

                      )}°`

                    : "--"}

                </strong>



                <small>

                  forecast

                </small>

              </div>



            </div>



            <div className="hourly-scroll">



              {hourly.map(

                (item, index) => (

                  <HourlyCard

                    key={item.time}

                    item={item}

                    active={index === 0}

                  />

                )

              )}



            </div>



            <HourlyChart

              hourly={hourly}

            />



          </div>



          {/* =================================================

              16 DAY

              ================================================= */}



          <div className="lower-section">



            <div className="section-header">



              <span className="section-kicker">

                EXTENDED OUTLOOK

              </span>



              <h3>

                16-day forecast

              </h3>



            </div>



            <div className="forecast-cards">



              {forecast

                .slice(0, 16)

                .map((day) => (

                  <div

                    className="forecast-card"

                    key={day.date}

                  >



                    <span>

                      {day.date?.slice(5)}

                    </span>



                    <div className="forecast-icon">

                      {weatherIcon(

                        day.weather_code,

                        true

                      )}

                    </div>



                    <strong>

                      {Math.round(

                        day.temperature_max ??

                          0

                      )}

                      °

                    </strong>



                    <small>

                      {Math.round(

                        day.temperature_min ??

                          0

                      )}

                      °

                    </small>



                    <div className="forecast-rain">

                      <CloudRain size={11} />



                      {Math.round(

                        day.rain_probability ??

                          0

                      )}

                      %

                    </div>



                  </div>

                ))}



            </div>



          </div>



        </section>

      )}



      {/* =================================================

          HAZARD DOCK

          ================================================= */}



      <HazardAlertDock

        risk={risk}

        maxRain={maxRain}

        maxRainHour={maxRainHour}

        onOpen={() =>

          hazardRef.current?.scrollIntoView({

            behavior: "smooth",

            block: "start",

          })

        }

      />



      {/* =================================================

          FOOTER

          ================================================= */}



      <footer className="footer">



        <div>

          <span>

            WEATHERSENSE

          </span>



          <small>

            Explainable ML Weather Forecasting &

            Pattern Intelligence

          </small>

        </div>



        <div className="footer-right">

          <span>LIVE DATA</span>

          <span>ML ENABLED</span>

          <span>24H ANALYTICS</span>

          <span>16D FORECAST</span>

        </div>



      </footer>



    </div>

  );

}