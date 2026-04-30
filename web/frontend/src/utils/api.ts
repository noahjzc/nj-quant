import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:10101/api',
});

export default api;
