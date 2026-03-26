import axios from 'axios';

describe('InferenceBridge API', () => {
	const baseUrl = process.env.INFBRIDGE_API_URL || 'http://localhost:8800';

	it('should be reachable', async () => {
		const resp = await axios.get(`${baseUrl}/v1/models`);
		expect(resp.status).toBe(200);
		expect(resp.data).toHaveProperty('data');
	});

	it('should list models', async () => {
		const resp = await axios.get(`${baseUrl}/v1/models`);
		expect(Array.isArray(resp.data.data)).toBe(true);
	});
});
